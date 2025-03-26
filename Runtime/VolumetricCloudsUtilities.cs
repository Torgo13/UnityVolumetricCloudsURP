using UnityEngine;

// Run in edit mode for easier testing
//[ExecuteInEditMode]
public class VolumetricCloudsUtilities : MonoBehaviour
{
    [Tooltip("The material of the volumetric clouds currently in use.")]
    // There's no shader checking, please provide the correct one
    [SerializeField] Material cloudsMaterial;

    // Global offset to the high frequency noise
    const float CLOUD_DETAIL_MIP_OFFSET = 0.0f;
    // Density below which we consider the density is zero (optimization reasons)
    const float CLOUD_DENSITY_TRESHOLD = 0.001f;
    // Number of steps before we start the large steps
    const int EMPTY_STEPS_BEFORE_LARGE_STEPS = 8;
    // Distance until which the erosion texture is used
    const float MIN_EROSION_DISTANCE = 3000.0f;
    const float MAX_EROSION_DISTANCE = 100000.0f;
    // Value that is used to normalize the noise textures
    const float NOISE_TEXTURE_NORMALIZATION_FACTOR = 100000.0f;
    // Maximal distance until the "skybox"
    const float MAX_SKYBOX_VOLUMETRIC_CLOUDS_DISTANCE = 200000.0f;

    const float FLT_MAX = float.MaxValue;

    // Dummy samplers
    const bool s_trilinear_repeat_sampler = true;
    const bool s_linear_repeat_sampler = true;

    // Readable Clouds Textures
    private Texture3D _ErosionNoise;
    private Texture3D _Worley128RGBA;
    private Texture2D _CloudCurveTexture;

    // Shader Keywords
    private bool _LOCAL_VOLUMETRIC_CLOUDS;
    private bool _CLOUDS_MICRO_EROSION;

    // Material Properties
    private const string localClouds = "_LOCAL_VOLUMETRIC_CLOUDS";
    private const string microErosion = "_CLOUDS_MICRO_EROSION";
    private static readonly int numPrimarySteps = Shader.PropertyToID("_NumPrimarySteps");
    private static readonly int maxStepSize = Shader.PropertyToID("_MaxStepSize");
    private static readonly int highestCloudAltitude = Shader.PropertyToID("_HighestCloudAltitude");
    private static readonly int lowestCloudAltitude = Shader.PropertyToID("_LowestCloudAltitude");
    private static readonly int shapeNoiseOffset = Shader.PropertyToID("_ShapeNoiseOffset");
    private static readonly int verticalShapeNoiseOffset = Shader.PropertyToID("_VerticalShapeNoiseOffset");
    private static readonly int globalOrientation = Shader.PropertyToID("_WindDirection");
    private static readonly int globalSpeed = Shader.PropertyToID("_WindVector");
    private static readonly int verticalShapeDisplacement = Shader.PropertyToID("_VerticalShapeWindDisplacement");
    private static readonly int verticalErosionDisplacement = Shader.PropertyToID("_VerticalErosionWindDisplacement");
    private static readonly int shapeSpeedMultiplier = Shader.PropertyToID("_MediumWindSpeed");
    private static readonly int erosionSpeedMultiplier = Shader.PropertyToID("_SmallWindSpeed");
    private static readonly int altitudeDistortion = Shader.PropertyToID("_AltitudeDistortion");
    private static readonly int densityMultiplier = Shader.PropertyToID("_DensityMultiplier");
    private static readonly int shapeScale = Shader.PropertyToID("_ShapeScale");
    private static readonly int shapeFactor = Shader.PropertyToID("_ShapeFactor");
    private static readonly int erosionScale = Shader.PropertyToID("_ErosionScale");
    private static readonly int erosionFactor = Shader.PropertyToID("_ErosionFactor");
    private static readonly int erosionOcclusion = Shader.PropertyToID("_ErosionOcclusion");
    private static readonly int microErosionScale = Shader.PropertyToID("_MicroErosionScale");
    private static readonly int microErosionFactor = Shader.PropertyToID("_MicroErosionFactor");
    private static readonly int fadeInStart = Shader.PropertyToID("_FadeInStart");
    private static readonly int fadeInDistance = Shader.PropertyToID("_FadeInDistance");
    private static readonly int earthRadius = Shader.PropertyToID("_EarthRadius");
    private static readonly int erosionNoise = Shader.PropertyToID("_ErosionNoise");
    private static readonly int worleyNoise = Shader.PropertyToID("_Worley128RGBA");
    private static readonly int cloudsCurveLut = Shader.PropertyToID("_CloudCurveTexture");

    private float _EarthRadius = 6378100.0f;
    private float _LowestCloudAltitude = 1200.0f;
    private float _HighestCloudAltitude = 3200.0f;
    private float _NumPrimarySteps = 32.0f;
    private float _MaxStepSize = 250.0f;
    private float _FadeInStart = 0.0f;
    private float _FadeInDistance = 5000.0f;
    private Vector4 _WindDirection = new Vector4(0.0f, 0.0f, 0.0f, 0.0f);
    private Vector4 _WindVector = new Vector4(0.0f, 0.0f, 0.0f, 0.0f);
    private Vector4 _ShapeNoiseOffset = new Vector4(0.0f, 0.0f, 0.0f, 0.0f);
    private float _VerticalShapeNoiseOffset = 0.0f;
    private float _VerticalShapeWindDisplacement = 0.0f;
    private float _VerticalErosionWindDisplacement = 0.0f;
    private float _MediumWindSpeed = 0.0f;
    private float _SmallWindSpeed = 0.0f;
    private float _ShapeScale = 0.0f;
    private float _ShapeFactor = 0.0f;
    private float _ErosionScale = 0.0f;
    private float _ErosionFactor = 0.0f;
    private float _ErosionOcclusion = 0.0f;
    private float _MicroErosionScale = 0.0f;
    private float _MicroErosionFactor = 0.0f;
    private float _DensityMultiplier = 0.0f;
    private float _AltitudeDistortion = 0.0f;
    //private Vector3 _WorldSpaceCameraPos = new(0.0f, 0.0f, 0.0f);
    private Vector3 _PlanetCenterPosition = new(0.0f, 0.0f, 0.0f);

    // HLSL Functions in C#
    private Vector3 ConvertToPS(Vector3 x) => (x - _PlanetCenterPosition);
    private float dot(Vector3 x, Vector3 y) => Vector3.Dot(x, y);
    private float exp(float value) => Mathf.Exp(value);
    private float lerp(float a, float b, float t) => Mathf.Lerp(a, b, t);
    private float length(Vector3 value) => Vector3.Distance(value, Vector3.zero);
    private float RangeRemap(float min, float max, float t) => saturate((t - min) / (max - min));
    private float sign(float value) => Mathf.Sign(value);
    private float Sq(float value) => value * value;
    private float sqrt(float value) => Mathf.Sqrt(value);
    private float saturate(float value) => Mathf.Clamp01(value);
    private float min(float x, float y) => Mathf.Min(x, y);
    private float max(float x, float y) => Mathf.Max(x, y);
    private Vector4 SAMPLE_TEXTURE3D_LOD(Texture3D tex3D, bool dummySampler, Vector3 texCoord, float dummyLod) => tex3D.GetPixelBilinear(texCoord.x, texCoord.y, texCoord.z);
    private Vector4 SAMPLE_TEXTURE2D_LOD(Texture2D tex2D, bool dummySampler, Vector2 texCoord, float dummyLod) => tex2D.GetPixelBilinear(texCoord.x, texCoord.y);

    struct Ray
    {
        // Origin of the ray in world space
        public Vector3 originWS;
        // Direction of the ray in world space
        public Vector3 direction;
        // Maximal ray length before hitting the far plane or an occluder
        public float maxRayLength;
        // Integration Noise
        public float integrationNoise;
    };

    struct RayHit
    {
        // Amount of lighting that comes from the clouds
        public Vector3 inScattering;
        // Transmittance through the clouds
        public float transmittance;
        // Mean distance of the clouds
        public float meanDistance;
        // Flag that defines if the ray is valid or not
        public bool invalidRay;
    };

    // Volumetric Clouds
    private Vector2 IntersectSphere(float sphereRadius, float cosChi,
                                    float radialDistance, float rcpRadialDistance)
    {
        // r_o = float2(0, r)
        // r_d = float2(sinChi, cosChi)
        // p_s = r_o + t * r_d
        //
        // R^2 = dot(r_o + t * r_d, r_o + t * r_d)
        // R^2 = ((r_o + t * r_d).x)^2 + ((r_o + t * r_d).y)^2
        // R^2 = t^2 + 2 * dot(r_o, r_d) + dot(r_o, r_o)
        //
        // t^2 + 2 * dot(r_o, r_d) + dot(r_o, r_o) - R^2 = 0
        //
        // Solve: t^2 + (2 * b) * t + c = 0, where
        // b = r * cosChi,
        // c = r^2 - R^2.
        //
        // t = (-2 * b + sqrt((2 * b)^2 - 4 * c)) / 2
        // t = -b + sqrt(b^2 - c)
        // t = -b + sqrt((r * cosChi)^2 - (r^2 - R^2))
        // t = -b + r * sqrt((cosChi)^2 - 1 + (R/r)^2)
        // t = -b + r * sqrt(d)
        // t = r * (-cosChi + sqrt(d))
        //
        // Why do we do this? Because it is more numerically robust.

        float d = Sq(sphereRadius * rcpRadialDistance) - saturate(1 - cosChi * cosChi);

        // Return the value of 'd' for debugging purposes.
        return (d < 0.0f) ? new Vector2(-1.0f, -1.0f) : (radialDistance * new Vector2(-cosChi - sqrt(d),
                                                      -cosChi + sqrt(d)));
    }

    private float ComputeCosineOfHorizonAngle(float rcpR)
    {
        float R = _EarthRadius;
        float sinHor = R * rcpR;
        return -sqrt(saturate(1.0f - sinHor * sinHor));
    }

    // Function that interects a ray with a sphere (optimized for very large sphere), returns up to two positives distances.

    // numSolutions: 0, 1 or 2 positive solves
    // startWS: rayOriginWS, might be camera positionWS
    // dir: normalized ray direction
    // radius: planet radius
    // result: the distance of hitPos, which means the value of solves
    private int RaySphereIntersection(Vector3 startWS, Vector3 dir, float radius, out Vector2 result)
    {
        Vector3 startPS = startWS + new Vector3(0.0f, _EarthRadius, 0.0f);
        float a = dot(dir, dir);
        float b = 2.0f * dot(dir, startPS);
        float c = dot(startPS, startPS) - (radius * radius);
        float d = (b * b) - 4.0f * a * c;
        result = new Vector2(0.0f, 0.0f);
        int numSolutions = 0;
        if (d >= 0.0f)
        {
            // Compute the values required for the solution eval
            float sqrtD = sqrt(d);
            float q = -0.5f * (b + sign(b) * sqrtD);
            result = new Vector2(c / q, q / a);
            // Remove the solutions we do not want
            numSolutions = 2;
            if (result.x < 0.0f)
            {
                numSolutions--;
                result.x = result.y;
            }
            if (result.y < 0.0f)
                numSolutions--;
        }
        // Return the number of solutions
        return numSolutions;
    }

    // Returns true if the ray exits the cloud volume (doesn't intersect earth)
    // The ray is supposed to start inside the volume
    private bool ExitCloudVolume(Vector3 originPS, Vector3 dir, float higherBoundPS, out float tExit)
    {
        // Given that we are inside the volume, we are guaranteed to exit at the outer bound
        float radialDistance = length(originPS);
        float rcpRadialDistance = 1.0f / radialDistance;
        float cosChi = dot(originPS, dir) * rcpRadialDistance;
        tExit = IntersectSphere(higherBoundPS, cosChi, radialDistance, rcpRadialDistance).y;

        // If the ray intersects the earth, then the sun is occluded by the earth
        return cosChi >= ComputeCosineOfHorizonAngle(rcpRadialDistance);
    }

    struct RayMarchRange
    {
        // The start of the range
        public float start;
        // The length of the range
        public float end;
    };

    // Returns true if the ray intersects the cloud volume
    // Outputs the entry and exit distance from the volume
    private bool IntersectCloudVolume(Vector3 originPS, Vector3 dir, float lowerBoundPS, float higherBoundPS, out float tEntry, out float tExit)
    {
        bool intersect;
        float radialDistance = length(originPS);
        float rcpRadialDistance = 1.0f / radialDistance;
        float cosChi = dot(originPS, dir) * rcpRadialDistance;
        Vector2 tInner = IntersectSphere(lowerBoundPS, cosChi, radialDistance, rcpRadialDistance);
        Vector2 tOuter = IntersectSphere(higherBoundPS, cosChi, radialDistance, rcpRadialDistance);

        if (tInner.x < 0.0f && tInner.y >= 0.0f) // Below the lower bound
        {
            // The ray starts at the intersection with the lower bound and ends at the intersection with the outer bound
            tEntry = tInner.y;
            tExit = tOuter.y;
            // We don't see the clouds if they are behind Earth
            intersect = cosChi >= ComputeCosineOfHorizonAngle(rcpRadialDistance);
        }
        else // Inside or above the cloud volume
        {
            // The ray starts at the intersection with the outer bound, or at 0 if we are inside
            // The ray ends at the lower bound if we hit it, at the outer bound otherwise
            tEntry = max(tOuter.x, 0.0f);
            tExit = tInner.x >= 0.0f ? tInner.x : tOuter.y;
            // We don't see the clouds if we don't hit the outer bound
            intersect = tOuter.y >= 0.0f;
        }

        return intersect;
    }

    private bool GetCloudVolumeIntersection(Vector3 originWS, Vector3 dir, out RayMarchRange rayMarchRange)
    {
        rayMarchRange = default;
        return IntersectCloudVolume(ConvertToPS(originWS), dir, _LowestCloudAltitude, _HighestCloudAltitude, out rayMarchRange.start, out rayMarchRange.end);
    }

    struct CloudProperties
    {
        // Normalized float that tells the "amount" of clouds that is at a given location
        public float density;
        // Ambient occlusion for the ambient probe
        public float ambientOcclusion;
        // Normalized value that tells us the height within the cloud volume (vertically)
        public float height;
        // Transmittance of the cloud
        public float sigmaT;
    };

    // Global attenuation of the density based on the camera distance
    private float DensityFadeValue(float distanceToCamera)
    {
        return saturate((distanceToCamera - _FadeInStart) / (_FadeInStart + _FadeInDistance));
    }

    // Evaluate the erosion mip offset based on the camera distance
    private float ErosionMipOffset(float distanceToCamera)
    {
        return lerp(0.0f, 4.0f, saturate((distanceToCamera - MIN_EROSION_DISTANCE) / (MAX_EROSION_DISTANCE - MIN_EROSION_DISTANCE)));
    }

    // Function that returns the normalized height inside the cloud layer
    private float EvaluateNormalizedCloudHeight(Vector3 positionPS)
    {
        return RangeRemap(_LowestCloudAltitude, _HighestCloudAltitude, length(positionPS));
    }

    // Animation of the cloud shape position
    private Vector3 AnimateShapeNoisePosition(Vector3 positionPS)
    {
        // We reduce the top-view repetition of the pattern
        positionPS.y += (positionPS.x / 3.0f + positionPS.z / 7.0f);
        // We add the contribution of the wind displacements
        return positionPS + new Vector3(_WindVector.x, 0.0f, _WindVector.y) * _MediumWindSpeed + new Vector3(0.0f, _VerticalShapeWindDisplacement, 0.0f);
        //return positionPS;
    }

    // Animation of the cloud erosion position
    private Vector3 AnimateErosionNoisePosition(Vector3 positionPS)
    {
        return positionPS + new Vector3(_WindVector.x, 0.0f, _WindVector.y) * _SmallWindSpeed + new Vector3(0.0f, _VerticalErosionWindDisplacement, 0.0f);
        //return positionPS;
    }

    // Structure that holds all the data used to define the cloud density of a point in space
    struct CloudCoverageData
    {
        // From a top-down view, in what proportions this pixel has clouds
        public float coverage;
        // From a top-down view, in what proportions this pixel has clouds
        public float rainClouds;
        // Value that allows us to request the cloudtype using the density
        public float cloudType;
        // Maximal cloud height
        public float maxCloudHeight;
    };

    // Function that evaluates the coverage data for a given point in planet space
    private void GetCloudCoverageData(Vector3 positionPS, out CloudCoverageData data)
    {
        // Convert the position into dome space and center the texture is centered above (0, 0, 0)
        //float2 normalizedPosition = AnimateCloudMapPosition(positionPS).xz / _NormalizationFactor * _CloudMapTiling.xy + _CloudMapTiling.zw - 0.5;
        //#if defined(CLOUDS_SIMPLE_PRESET)
        Vector4 cloudMapData = new Vector4(0.9f, 0.0f, 0.25f, 1.0f);
        //#else
        //float4 cloudMapData = SAMPLE_TEXTURE2D_LOD(_CloudMapTexture, s_linear_repeat_sampler, float2(normalizedPosition), 0);
        //#endif
        data.coverage = cloudMapData.x;
        data.rainClouds = cloudMapData.y;
        data.cloudType = cloudMapData.z;
        data.maxCloudHeight = cloudMapData.w;
    }

    // Density remapping function
    private float DensityRemap(float x, float a, float b, float c, float d)
    {
        return (((x - a) / (b - a)) * (d - c)) + c;
    }

    // Function that evaluates the cloud properties at a given absolute world space position
    private void EvaluateCloudProperties(Vector3 positionPS, float noiseMipOffset, float erosionMipOffset, bool cheapVersion, bool lightSampling,
                                out CloudProperties properties)
    {
        // Initialize all the values to 0 in case
        properties.density = 0.0f;
        properties.ambientOcclusion = 0.0f;
        properties.height = 0.0f;
        properties.sigmaT = 0.0f;

        //#ifndef CLOUDS_SIMPLE_PRESET
        // When using a cloud map, we cannot support the full planet due to UV issues
        //#endif

        // Remove global clouds below the horizon
        if (!_LOCAL_VOLUMETRIC_CLOUDS && positionPS.y < _EarthRadius)
            return;

        // By default the ambient occlusion is 1.0
        properties.ambientOcclusion = 1.0f;

        // Evaluate the normalized height of the position within the cloud volume
        properties.height = EvaluateNormalizedCloudHeight(positionPS);

        // When rendering in camera space, we still want horizontal scrolling
        /*
        if (!_LOCAL_VOLUMETRIC_CLOUDS)
        {
            positionPS.x += _WorldSpaceCameraPos.x;
            positionPS.z += _WorldSpaceCameraPos.z;
        }
        */

        // Evaluate the generic sampling coordinates
        Vector3 animatePosPS = AnimateShapeNoisePosition(positionPS);
        animatePosPS = new Vector3(animatePosPS.x, animatePosPS.z, animatePosPS.y) / NOISE_TEXTURE_NORMALIZATION_FACTOR;
        Vector3 baseNoiseSamplingCoordinates = animatePosPS * _ShapeScale - new Vector3(_ShapeNoiseOffset.x, _ShapeNoiseOffset.y, _VerticalShapeNoiseOffset);

        // Evaluate the coordinates at which the noise will be sampled and apply wind displacement
        baseNoiseSamplingCoordinates += _AltitudeDistortion * properties.height * new Vector3(_WindDirection.x, _WindDirection.y, 0.0f);

        // Read the low frequency Perlin-Worley and Worley noises
        float lowFrequencyNoise = SAMPLE_TEXTURE3D_LOD(_Worley128RGBA, s_trilinear_repeat_sampler, baseNoiseSamplingCoordinates, noiseMipOffset).x;

        // Evaluate the cloud coverage data for this position
        GetCloudCoverageData(positionPS, out CloudCoverageData cloudCoverageData);

        // If this region of space has no cloud coverage, exit right away
        if (cloudCoverageData.coverage <= CLOUD_DENSITY_TRESHOLD || cloudCoverageData.maxCloudHeight < properties.height)
            return;

        // Read from the LUT
        //#if defined(CLOUDS_SIMPLE_PRESET)
        Vector3 densityErosionAO = SAMPLE_TEXTURE2D_LOD(_CloudCurveTexture, s_linear_repeat_sampler, new Vector2(0.0f, properties.height), 0);
        //#else
        //half3 densityErosionAO = SAMPLE_TEXTURE2D_LOD(_CloudLutTexture, s_linear_repeat_sampler, float2(cloudCoverageData.cloudType, properties.height), CLOUD_LUT_MIP_OFFSET).xyz;
        //#endif

        // Adjust the shape and erosion factor based on the LUT and the coverage
        float shapeFactor = lerp(0.1f, 1.0f, _ShapeFactor) * densityErosionAO.y;
        float erosionFactor = _ErosionFactor * densityErosionAO.y;
        float microDetailFactor = 0.0f;
        if (_CLOUDS_MICRO_EROSION)
            microDetailFactor = _MicroErosionFactor * densityErosionAO.y;

        // Combine with the low frequency noise, we want less shaping for large clouds
        lowFrequencyNoise = lerp(1.0f, lowFrequencyNoise, shapeFactor);
        float base_cloud = 1.0f - densityErosionAO.x * cloudCoverageData.coverage * (1.0f - shapeFactor);

        base_cloud = saturate(DensityRemap(lowFrequencyNoise, base_cloud, 1.0f, 0.0f, 1.0f)) * cloudCoverageData.coverage * cloudCoverageData.coverage;

        // Weight the ambient occlusion's contribution
        properties.ambientOcclusion = densityErosionAO.z;

        // Change the sigma based on the rain cloud data
        properties.sigmaT = lerp(0.04f, 0.12f, cloudCoverageData.rainClouds);

        // The ambient occlusion value that is baked is less relevant if there is shaping or erosion, small hack to compensate that
        float ambientOcclusionBlend = saturate(1.0f - max(erosionFactor, shapeFactor) * 0.5f);
        properties.ambientOcclusion = lerp(1.0f, properties.ambientOcclusion, ambientOcclusionBlend);

        // Apply the erosion for nicer details
        if (!cheapVersion)
        {
            //float erosionMipOffset = 0.5f;
            Vector3 erosionCoords = AnimateErosionNoisePosition(positionPS) / (NOISE_TEXTURE_NORMALIZATION_FACTOR * _ErosionScale);
            float erosionNoise = 1.0f - SAMPLE_TEXTURE3D_LOD(_ErosionNoise, s_linear_repeat_sampler, erosionCoords, CLOUD_DETAIL_MIP_OFFSET + erosionMipOffset).x;
            erosionNoise = lerp(0.0f, erosionNoise, erosionFactor * 0.75f * cloudCoverageData.coverage);
            properties.ambientOcclusion = saturate(properties.ambientOcclusion - sqrt(erosionNoise * _ErosionOcclusion));
            base_cloud = DensityRemap(base_cloud, erosionNoise, 1.0f, 0.0f, 1.0f);

            if (_CLOUDS_MICRO_EROSION)
            {
                Vector3 fineCoords = AnimateErosionNoisePosition(positionPS) / (NOISE_TEXTURE_NORMALIZATION_FACTOR * _MicroErosionScale);
                float fineNoise = 1.0f - SAMPLE_TEXTURE3D_LOD(_ErosionNoise, s_linear_repeat_sampler, fineCoords, CLOUD_DETAIL_MIP_OFFSET + erosionMipOffset).x;
                fineNoise = lerp(0.0f, fineNoise, microDetailFactor * 0.5f * cloudCoverageData.coverage);
                base_cloud = DensityRemap(base_cloud, fineNoise, 1.0f, 0.0f, 1.0f);
            }
        }

        // Make sure we do not send any negative values
        base_cloud = max(0.0f, base_cloud);

        // Attenuate everything by the density multiplier
        properties.density = base_cloud * _DensityMultiplier;        
    }

    private RayHit TraceCloudsRay(in Ray ray)
    {
        RayHit rayHit;
        rayHit.inScattering = new(0.0f, 0.0f, 0.0f);
        rayHit.transmittance = 1.0f;
        rayHit.meanDistance = FLT_MAX;
        rayHit.invalidRay = true;

        // Determine if ray intersects bounding volume, if the ray does not intersect the cloud volume AABB, skip right away
        if (GetCloudVolumeIntersection(ray.originWS, ray.direction, out RayMarchRange rayMarchRange))
        {
            if (ray.maxRayLength >= rayMarchRange.start)
            {
                // Initialize the depth for accumulation
                rayHit.meanDistance = 0.0f;

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
                //Vector3 rayMarchStartPS = ConvertToPS(ray.originWS) + rayMarchRange.start * ray.direction;
                //Vector3 rayMarchEndPS = rayMarchStartPS + totalDistance * ray.direction;

                // Tracking the number of steps that have been made
                int currentIndex = 0;

                // Normalization value of the depth
                float meanDistanceDivider = 0.0f;

                // Current position for the evaluation, apply blue noise to start position
                float currentDistance = ray.integrationNoise;
                Vector3 currentPositionWS = ray.originWS + (rayMarchRange.start + currentDistance) * ray.direction;

                // Initialize the values for the optimized ray marching
                bool activeSampling = true;
                int sequentialEmptySamples = 0;

                // Do the ray march for every step that we can.
                while (currentIndex < (int)_NumPrimarySteps && currentDistance < totalDistance)
                {
                    // Compute the camera-distance based attenuation
                    float densityAttenuationValue = DensityFadeValue(rayMarchRange.start + currentDistance);
                    // Compute the mip offset for the erosion texture
                    float erosionMipOffset = ErosionMipOffset(rayMarchRange.start + currentDistance);

                    // Should we be evaluating the clouds or just doing the large ray marching
                    if (activeSampling)
                    {
                        // Convert to planet space
                        Vector3 positionPS = ConvertToPS(currentPositionWS);

                        // If the density is null, we can skip as there will be no contribution
                        EvaluateCloudProperties(positionPS, 0.0f, erosionMipOffset, false, false, out CloudProperties properties);

                        // Apply the fade in function to the density
                        properties.density *= densityAttenuationValue;

                        if (properties.density > CLOUD_DENSITY_TRESHOLD)
                        {
                            // Contribute to the average depth (must be done first in case we end up inside a cloud at the next step)
                            float transmitanceXdensity = rayHit.transmittance * properties.density;
                            rayHit.meanDistance += (rayMarchRange.start + currentDistance) * transmitanceXdensity;
                            meanDistanceDivider += transmitanceXdensity;

                            // Evaluate the cloud at the position
                            //EvaluateCloud(properties, ray.direction, currentPositionWS, rayMarchStartPS, rayMarchEndPS, stepS, currentDistance / totalDistance, rayHit);
                            // No lighting Version
                            {
                                float extinction = properties.density * properties.sigmaT;
                                float transmittance = exp(-extinction * stepS);
                                rayHit.transmittance *= transmittance;
                            }

                            // if most of the energy is absorbed, just leave.
                            if (rayHit.transmittance < 0.003f)
                            {
                                rayHit.transmittance = 0.0f;
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
                        float relativeStepSize = lerp(ray.integrationNoise, 1.0f, saturate(currentIndex));
                        currentPositionWS += relativeStepSize * stepS * ray.direction;
                        currentDistance += stepS * relativeStepSize;

                    }
                    else
                    {
                        // Convert to planet space
                        Vector3 positionPS = ConvertToPS(currentPositionWS);

                        EvaluateCloudProperties(positionPS, 1.0f, 0.0f, true, false, out CloudProperties properties);

                        // Apply the fade in function to the density
                        properties.density *= densityAttenuationValue;

                        // If the density is lower than our tolerance,
                        if (properties.density < CLOUD_DENSITY_TRESHOLD)
                        {
                            currentPositionWS += 2.0f * stepS * ray.direction;
                            currentDistance += stepS * 2.0f;
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
                if (rayHit.meanDistance == 0.0f)
                    rayHit.invalidRay = true;
                else
                {
                    rayHit.meanDistance /= meanDistanceDivider;
                    rayHit.invalidRay = false;
                }
            }
        }
        return rayHit;
    }

    private void UpdateCloudsProperties()
    {
        _ErosionNoise = (Texture3D)cloudsMaterial.GetTexture(erosionNoise);
        _Worley128RGBA = (Texture3D)cloudsMaterial.GetTexture(worleyNoise);
        _CloudCurveTexture = (Texture2D)cloudsMaterial.GetTexture(cloudsCurveLut);

        _LOCAL_VOLUMETRIC_CLOUDS = cloudsMaterial.IsKeywordEnabled(localClouds);
        _CLOUDS_MICRO_EROSION = cloudsMaterial.IsKeywordEnabled(microErosion);

        _NumPrimarySteps = cloudsMaterial.GetFloat(numPrimarySteps);
        _MaxStepSize = cloudsMaterial.GetFloat(maxStepSize);
        _HighestCloudAltitude = cloudsMaterial.GetFloat(highestCloudAltitude);
        _LowestCloudAltitude = cloudsMaterial.GetFloat(lowestCloudAltitude);
        _ShapeNoiseOffset = cloudsMaterial.GetVector(shapeNoiseOffset);
        _VerticalShapeNoiseOffset = cloudsMaterial.GetFloat(verticalShapeNoiseOffset);

        _WindDirection = cloudsMaterial.GetVector(globalOrientation);
        _WindVector = cloudsMaterial.GetVector(globalSpeed);
        _MediumWindSpeed = cloudsMaterial.GetFloat(shapeSpeedMultiplier);
        _SmallWindSpeed = cloudsMaterial.GetFloat(erosionSpeedMultiplier);
        _AltitudeDistortion = cloudsMaterial.GetFloat(altitudeDistortion);
        _VerticalShapeWindDisplacement = cloudsMaterial.GetFloat(verticalShapeDisplacement);
        _VerticalErosionWindDisplacement = cloudsMaterial.GetFloat(verticalErosionDisplacement);

        _DensityMultiplier = cloudsMaterial.GetFloat(densityMultiplier);
        _ShapeScale = cloudsMaterial.GetFloat(shapeScale);
        _ShapeFactor = cloudsMaterial.GetFloat(shapeFactor);
        _ErosionScale = cloudsMaterial.GetFloat(erosionScale);
        _ErosionFactor = cloudsMaterial.GetFloat(erosionFactor);
        _ErosionOcclusion = cloudsMaterial.GetFloat(erosionOcclusion);
        _MicroErosionScale = cloudsMaterial.GetFloat(microErosionScale);
        _MicroErosionFactor = cloudsMaterial.GetFloat(microErosionFactor);
    
        _FadeInStart = cloudsMaterial.GetFloat(fadeInStart);
        _FadeInDistance = cloudsMaterial.GetFloat(fadeInDistance);
        _EarthRadius = cloudsMaterial.GetFloat(earthRadius);
        _PlanetCenterPosition = new Vector3(0.0f, -_EarthRadius, 0.0f);
    }

    /// <summary>
    /// Calculates the density of volumetric clouds on the CPU along a ray from a given start position in world space.
    /// </summary>
    /// <param name="startPosWS">The start position of the ray in world space.</param>
    /// <param name="directionWS">The normalized direction of the ray in world space.</param>
    /// <returns>
    /// The cloud density along the ray.
    /// </returns>
    public float QueryCloudsRay(Vector3 startPosWS, Vector3 directionWS)
    {
        if (cloudsMaterial == null)
            return 0.0f;

        UpdateCloudsProperties();

        Ray ray;
        ray.originWS = startPosWS;
        ray.direction = directionWS;
        ray.maxRayLength = MAX_SKYBOX_VOLUMETRIC_CLOUDS_DISTANCE;
        ray.integrationNoise = 0.0f;

        RayHit rayHit = TraceCloudsRay(in ray);

        return rayHit.invalidRay ? 0.0f : 1.0f - rayHit.transmittance;
    }

    // An example of query clouds density
    /*private void Update()
    {
        Vector3 startPosWS = new Vector3(0.0f, 0.0f, 0.0f);
        Vector3 directionWS = new Vector3(0.0f, 1.0f, 0.0f); // make sure it's normalized
        float density = QueryCloudsRay(startPosWS , directionWS);

        Debug.Log(density);
    }*/
}
