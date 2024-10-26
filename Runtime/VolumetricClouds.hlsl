#ifndef URP_VOLUMETRIC_CLOUDS_HLSL
#define URP_VOLUMETRIC_CLOUDS_HLSL

#include "./VolumetricCloudsDefs.hlsl"
#include "./VolumetricCloudsUtilities.hlsl"

Ray BuildCloudsRay(float2 screenUV, float depth, half3 invViewDirWS, bool isOccluded)
{
    Ray ray;

#ifdef _LOCAL_VOLUMETRIC_CLOUDS
    ray.originWS = GetCameraPositionWS();
#else
    ray.originWS = float3(0.0, 0.0, 0.0);
#endif

    ray.direction = invViewDirWS;

    // Compute the max cloud ray length
    // For opaque objects, we only care about clouds in front of them.
#ifdef _LOCAL_VOLUMETRIC_CLOUDS
    // The depth may from a high-res texture which isn't ideal but can save performance.
    float distance = LinearEyeDepth(depth, _ZBufferParams) * rcp(dot(ray.direction, -UNITY_MATRIX_V[2].xyz));
    ray.maxRayLength = lerp(MAX_SKYBOX_VOLUMETRIC_CLOUDS_DISTANCE, distance, isOccluded);
#else
    ray.maxRayLength = MAX_SKYBOX_VOLUMETRIC_CLOUDS_DISTANCE;
#endif

    ray.integrationNoise = GenerateRandomFloat(screenUV);

    return ray;
}

RayHit TraceCloudsRay(Ray ray)
{
    RayHit rayHit;
    rayHit.inScattering = half3(0.0, 0.0, 0.0);
    rayHit.transmittance = 1.0;
    rayHit.meanDistance = FLT_MAX;
    rayHit.invalidRay = true;

    // Determine if ray intersects bounding volume, if the ray does not intersect the cloud volume AABB, skip right away
    RayMarchRange rayMarchRange;
    if (GetCloudVolumeIntersection(ray.originWS, ray.direction, rayMarchRange))
    {
        if (ray.maxRayLength >= rayMarchRange.start)
        {
            // Initialize the depth for accumulation
            rayHit.meanDistance = 0.0;

            // Total distance that the ray must travel including empty spaces
            // Clamp the travel distance to whatever is closer
            // - Sky Occluder
            // - Volume end
            // - Far plane
            float totalDistance = min(rayMarchRange.end, ray.maxRayLength) - rayMarchRange.start;

            // Evaluate our integration step
            float stepS = min(totalDistance / (float)_NumPrimarySteps, _MaxStepSize);
            totalDistance = stepS * _NumPrimarySteps;

            // Compute the environment lighting that is going to be used for the cloud evaluation
            float3 rayMarchStartPS = ConvertToPS(ray.originWS) + rayMarchRange.start * ray.direction;
            float3 rayMarchEndPS = rayMarchStartPS + totalDistance * ray.direction;

            // Tracking the number of steps that have been made
            int currentIndex = 0;

            // Normalization value of the depth
            float meanDistanceDivider = 0.0;

            // Current position for the evaluation, apply blue noise to start position
            float currentDistance = ray.integrationNoise;
            float3 currentPositionWS = ray.originWS + (rayMarchRange.start + currentDistance) * ray.direction;

            // Initialize the values for the optimized ray marching
            bool activeSampling = true;
            int sequentialEmptySamples = 0;

            // Do the ray march for every step that we can.
            while (currentIndex < (int)_NumPrimarySteps && currentDistance < totalDistance)
            {
                // Convert to planet space
                float3 positionPS = ConvertToPS(currentPositionWS);

                // Only check for terrain below the maximum terrain height
                if (_TerrainData.z > half(0.0) && currentPositionWS.y < _TerrainData.y)
                {
                    activeSampling = true;

                    half4 terrainProperties;
                    EvaluateTerrainProperties(positionPS, terrainProperties);

                    if (currentPositionWS.y < terrainProperties.w)
                    {
                        // Refine hit position using binary search
                        float3 wpos;
                        float relativeRayDistance = currentDistance / 512.0;
                        float t1 = currentDistance; // Current distance
                        float t0 = currentDistance - stepS; // Previous distance
                        half ao = half(1.0);
                        for (int i = 0; i < _TerrainData.w; i++) {
                            currentDistance = (t1 + t0) * 0.5; // Midpoint between previous and current distance
                            currentPositionWS = ray.originWS + (rayMarchRange.start + currentDistance) * ray.direction;
                            wpos = floor(currentPositionWS) + 0.5;
                            EvaluateTerrainProperties(ConvertToPS(wpos), terrainProperties);
                            if (wpos.y < terrainProperties.w) {
                                t1 = currentDistance;
                                relativeRayDistance = currentDistance / 512.0;
                                ao = saturate(0.25 + frac(currentPositionWS.y) * 0.75 + relativeRayDistance);
                            } else {
                                t0 = currentDistance;
                            }
                        }
                        
                        half atten = half(1.0);

                        //if (_ShadowIntensity > half(0.0))
                        {
                            half4 terrainShadowProperties;
                            const half incr = half(1.015);
                            for (float j = 2.0; j < _TerrainData.y; j = j * incr + incr) {
                                float3 rpos = currentPositionWS + _MainLightPosition.xyz * j;
                                if (rpos.y > _TerrainData.y) {
                                    break; // Above terrain max altitude so in direct light
                                }
                    
                                EvaluateTerrainProperties(ConvertToPS(rpos), terrainShadowProperties);
                                if (rpos.y < terrainShadowProperties.w) {
                                    atten = _ShadowIntensity;
                                    break;
                                }
                            }
                        }

                        rayHit.meanDistance = currentDistance;
                        meanDistanceDivider = 1.0;
                        
                        // Evaluate the terrain at the position
                        atten = saturate(saturate(atten + _MainLightPosition.y * _VPDaylightShadowAtten) + _VPAmbientLight);
                        EvaluateTerrain(terrainProperties, ray.direction, currentPositionWS, rayMarchStartPS, rayMarchEndPS, saturate(ao * atten), stepS, relativeRayDistance, rayHit);
                        break;
                    }
                }

                // Compute the camera-distance based attenuation
                float densityAttenuationValue = DensityFadeValue(rayMarchRange.start + currentDistance);

                // Should we be evaluating the clouds or just doing the large ray marching
                if (activeSampling)
                {
                    // Compute the mip offset for the erosion texture
                    float erosionMipOffset = ErosionMipOffset(rayMarchRange.start + currentDistance);

                    // If the density is null, we can skip as there will be no contribution
                    CloudProperties properties;
                    EvaluateCloudProperties(positionPS, 0.0, erosionMipOffset, false, false, properties);

                    // Apply the fade in function to the density
                    properties.density *= densityAttenuationValue;

                    if (properties.density > CLOUD_DENSITY_TRESHOLD)
                    {
                        // Contribute to the average depth (must be done first in case we end up inside a cloud at the next step)
                        half transmitanceXdensity = rayHit.transmittance * properties.density;
                        rayHit.meanDistance += (rayMarchRange.start + currentDistance) * transmitanceXdensity;
                        meanDistanceDivider += transmitanceXdensity;

                        // Evaluate the cloud at the position
                        EvaluateCloud(properties, ray.direction, currentPositionWS, rayMarchStartPS, rayMarchEndPS, stepS, currentDistance / totalDistance, rayHit);

                        // if most of the energy is absorbed, just leave.
                        if (rayHit.transmittance < 0.003)
                        {
                            rayHit.transmittance = 0.0;
                            break;
                        }

                        // Reset the empty sample counter
                        sequentialEmptySamples = 0;
                    }
                    else
                        sequentialEmptySamples++;

                    // If it has been more than EMPTY_STEPS_BEFORE_LARGE_STEPS, disable active sampling and start large steps
                    if (sequentialEmptySamples == EMPTY_STEPS_BEFORE_LARGE_STEPS)
                        activeSampling = false;

                    // Do the next step
                    float relativeStepSize = lerp(ray.integrationNoise, 1.0, saturate(currentIndex));
                    currentPositionWS += ray.direction * stepS * relativeStepSize;
                    currentDistance += stepS * relativeStepSize;
                }
                else
                {
                    CloudProperties properties;
                    EvaluateCloudProperties(positionPS, 1.0, 0.0, true, false, properties);

                    // Apply the fade in function to the density
                    properties.density *= densityAttenuationValue;

                    // If the density is lower than our tolerance,
                    if (properties.density < CLOUD_DENSITY_TRESHOLD)
                    {
                        currentPositionWS += ray.direction * stepS * 2.0;
                        currentDistance += stepS * 2.0;
                    }
                    else 
                    {
                        // Somewhere between this step and the previous clouds started
                        // We reset all the counters and enable active sampling
                        currentPositionWS -= ray.direction * stepS;
                        currentDistance -= stepS;
                        activeSampling = true;
                        sequentialEmptySamples = 0;
                    }
                }

                currentIndex++;
            }

            // Normalized the depth we computed
            if (rayHit.meanDistance > 0.0)
            {
                rayHit.meanDistance /= meanDistanceDivider;
                rayHit.invalidRay = false;
            }
        }
    }
    return rayHit;
}

#endif