#define _LOCAL_VOLUMETRIC_CLOUDS
#define _CONST_EARTH_RADIUS

using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;

// Run in edit mode for easier testing
//[ExecuteInEditMode]
public class VolumetricCloudsUtilities
{
    public VolumetricCloudsUtilities(Material cloudsMaterial)
    {
        this.cloudsMaterial = cloudsMaterial;
        UpdateCloudsProperties();
    }

    [Tooltip("The material of the volumetric clouds currently in use.")]
    /// <summary>There's no shader checking, please provide the correct one</summary>
    //[SerializeField]
    readonly
    Material cloudsMaterial;

    /// <summary>Global offset to the high frequency noise</summary>
    const float CLOUD_DETAIL_MIP_OFFSET = 0.0f;
    /// <summary>Density below which we consider the density is zero (optimization reasons)</summary>
    const float CLOUD_DENSITY_TRESHOLD = 0.001f;
    /// <summary>Number of steps before we start the large steps</summary>
    const int EMPTY_STEPS_BEFORE_LARGE_STEPS = 8;
    // Distance until which the erosion texture is used
    const float MIN_EROSION_DISTANCE = 3000.0f;
    const float MAX_EROSION_DISTANCE = 100000.0f;
    /// <summary>Value that is used to normalize the noise textures</summary>
    const float NOISE_TEXTURE_NORMALIZATION_FACTOR = 100000.0f;
    /// <summary>Maximal distance until the "skybox"</summary>
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
#if _CONST_EARTH_RADIUS
#else
    private static readonly int earthRadius = Shader.PropertyToID("_EarthRadius");
#endif // _CONST_EARTH_RADIUS
    private static readonly int erosionNoise = Shader.PropertyToID("_ErosionNoise");
    private static readonly int worleyNoise = Shader.PropertyToID("_Worley128RGBA");
    private static readonly int cloudsCurveLut = Shader.PropertyToID("_CloudCurveTexture");

#if _CONST_EARTH_RADIUS
    private const float _EarthRadius = 6378100.0f;
#else
    private float _EarthRadius = 6378100.0f;
#endif // _CONST_EARTH_RADIUS
    private float _LowestCloudAltitude = 1200.0f;
    private float _HighestCloudAltitude = 3200.0f;
    private float _NumPrimarySteps = 32.0f;
    private float _MaxStepSize = 250.0f;
    private float _FadeInStart = 0.0f;
    private float _FadeInDistance = 5000.0f;
    private float4 _WindDirection = new float4(0.0f, 0.0f, 0.0f, 0.0f);
    private float4 _WindVector = new float4(0.0f, 0.0f, 0.0f, 0.0f);
    private float4 _ShapeNoiseOffset = new float4(0.0f, 0.0f, 0.0f, 0.0f);
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
    //private float3 _WorldSpaceCameraPos = new(0.0f, 0.0f, 0.0f);
    private float3 _PlanetCenterPosition = new(0.0f, 0.0f, 0.0f);

    // HLSL Functions in C#
    private float3 ConvertToPS(float3 x) => (x - _PlanetCenterPosition);
    private static float RangeRemap(float min, float max, float t) => saturate((t - min) / (max - min));
    private static float Sq(float value) => value * value;
    private static float SAMPLE_TEXTURE3D_LOD(Texture3D tex3D, bool dummySampler, float3 texCoord, float dummyLod) => tex3D.GetPixelBilinear(texCoord.x, texCoord.y, texCoord.z).r;
    private static float4 SAMPLE_TEXTURE2D_LOD(Texture2D tex2D, bool dummySampler, float2 texCoord, float dummyLod) => (Vector4)tex2D.GetPixelBilinear(texCoord.x, texCoord.y);

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct CloudRay
    {
        /// <summary>Origin of the ray in world space</summary>
        public
        float3 originWS;
        /// <summary>Maximal ray length before hitting the far plane or an occluder</summary>
        public
        float maxRayLength;
        /// <summary>Direction of the ray in world space</summary>
        public
        float3 direction;
        /// <summary>Integration Noise</summary>
        public
        float integrationNoise;
    };

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct VolumetricRayResult
    {
        /// <summary>Amount of lighting that comes from the clouds</summary>
        public
        float3 scattering;
        /// <summary>Transmittance through the clouds</summary>
        public
        float transmittance;
        /// <summary>Mean distance of the clouds</summary>
        public
        float meanDistance;
        /// <summary>Flag that defines if the ray is valid or not</summary>
        [System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.U1)]
        public
        bool invalidRay;
    };

    // Volumetric Clouds
    static
    float2 IntersectSphere(float sphereRadius, float cosChi,
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
        return (d < 0) ? d : (radialDistance * float2(-cosChi - sqrt(d),
                                                        -cosChi + sqrt(d)));
    }

#if _CONST_EARTH_RADIUS
    static
#endif // _CONST_EARTH_RADIUS
    float ComputeCosineOfHorizonAngle(float r)
    {
        float R = _EarthRadius;
        float sinHor = R * rcp(r);
        return -sqrt(saturate(1 - sinHor * sinHor));
    }

#if UNUSED
    // Function that interects a ray with a sphere (optimized for very large sphere), returns up to two positives distances.

    // numSolutions: 0, 1 or 2 positive solves
    // startWS: rayOriginWS, might be camera positionWS
    // dir: normalized ray direction
    // radius: planet radius
    // result: the distance of hitPos, which means the value of solves
    int RaySphereIntersection(float3 startWS, float3 dir, float radius, out float2 result)
    {
        float3 startPS = startWS + float3(0, _EarthRadius, 0);
        float a = dot(dir, dir);
        float b = 2.0f * dot(dir, startPS);
        float c = dot(startPS, startPS) - (radius * radius);
        float d = (b * b) - 4.0f * a * c;
        result = default;
        int numSolutions = 0;
        if (d >= 0.0f)
        {
            // Compute the values required for the solution eval
            float sqrtD = sqrt(d);
            float q = -0.5f * (b + sign(b) * sqrtD);
            result = float2(c / q, q / a);
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
    bool ExitCloudVolume(float3 originPS, half3 dir, float higherBoundPS, out float tExit)
    {
        // Given that we are inside the volume, we are guaranteed to exit at the outer bound
        float radialDistance = length(originPS);
        float cosChi = dot(originPS, dir) * rcp(radialDistance);
        tExit = IntersectSphere(higherBoundPS, cosChi, radialDistance, rcp(radialDistance)).y;

        // If the ray intersects the earth, then the sun is occluded by the earth
        return cosChi >= ComputeCosineOfHorizonAngle(radialDistance);
    }
#endif // UNUSED

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct RayMarchRange
    {
        /// <summary>The start of the range</summary>
        public
        float start;
        /// <summary>The length of the range</summary>
        public
        float end;
    };

    /// <summary>Returns true if the ray intersects the cloud volume</summary>
    /// <returns>Outputs the entry and exit distance from the volume</returns>
#if _CONST_EARTH_RADIUS
    static
#endif // _CONST_EARTH_RADIUS
    bool IntersectCloudVolume(float3 originPS, half3 dir, float lowerBoundPS, float higherBoundPS, out float tEntry, out float tExit)
    {
        bool intersect;
        float radialDistance = length(originPS);
        float rcpRadialDistance = rcp(radialDistance);
        float cosChi = dot(originPS, dir) * rcpRadialDistance;
        float2 tInner = IntersectSphere(lowerBoundPS, cosChi, radialDistance, rcpRadialDistance);
        float2 tOuter = IntersectSphere(higherBoundPS, cosChi, radialDistance, rcpRadialDistance);

        if (tInner.x < 0.0f && tInner.y >= 0.0f) // Below the lower bound
        {
            // The ray starts at the intersection with the lower bound and ends at the intersection with the outer bound
            tEntry = tInner.y;
            tExit = tOuter.y;
            // We don't see the clouds if they are behind Earth
            intersect = cosChi >= ComputeCosineOfHorizonAngle(radialDistance);
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

    bool GetCloudVolumeIntersection(float3 originWS, half3 dir, out RayMarchRange rayMarchRange)
    {
#if _LOCAL_VOLUMETRIC_CLOUDS
        return IntersectCloudVolume(ConvertToPS(originWS), dir, _LowestCloudAltitude, _HighestCloudAltitude, out rayMarchRange.start, out rayMarchRange.end);
#else
        {
            ZERO_INITIALIZE(RayMarchRange, rayMarchRange);

            // intersect with all three spheres
            float2 intersectionInter, intersectionOuter;
            int numInterInner = RaySphereIntersection(originWS, dir, _LowestCloudAltitude, intersectionInter);
            int numInterOuter = RaySphereIntersection(originWS, dir, _HighestCloudAltitude, intersectionOuter);

            // The ray starts at the first intersection with the lower bound and goes up to the first intersection with the outer bound
            rayMarchRange.start = intersectionInter.x;
            rayMarchRange.end = intersectionOuter.x;

            // Return if we have an intersection
            return true;
        }
#endif
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct CloudProperties
    {
        /// <summary>Normalized float that tells the "amount" of clouds that is at a given location</summary>
        public
        float density;
        /// <summary>Ambient occlusion for the ambient probe</summary>
        public
        float ambientOcclusion;
        /// <summary>Normalized value that tells us the height within the cloud volume (vertically)</summary>
        public
        float height;
        /// <summary>Transmittance of the cloud</summary>
        public
        float sigmaT;
    };

    /// <summary>Global attenuation of the density based on the camera distance</summary>
    float DensityFadeValue(float distanceToCamera)
    {
        return DensityFadeValue(distanceToCamera, _FadeInStart, _FadeInDistance);
    }

    /// <inheritdoc cref="DensityFadeValue(float)"/>
    static
    float DensityFadeValue(float distanceToCamera, float _FadeInStart, float _FadeInDistance)
    {
        return saturate((distanceToCamera - _FadeInStart) * rcp(_FadeInStart + _FadeInDistance));
    }

    /// <summary>Evaluate the erosion mip offset based on the camera distance</summary>
    static
    float ErosionMipOffset(float distanceToCamera)
    {
        return lerp(0.0f, 4.0f, saturate((distanceToCamera - MIN_EROSION_DISTANCE) * rcp(MAX_EROSION_DISTANCE - MIN_EROSION_DISTANCE)));
    }

    /// <summary>Function that returns the normalized height inside the cloud layer</summary>
    float EvaluateNormalizedCloudHeight(float3 positionPS)
    {
        return EvaluateNormalizedCloudHeight(positionPS, _LowestCloudAltitude, _HighestCloudAltitude);
    }

    /// <inheritdoc cref="EvaluateNormalizedCloudHeight(Unity.Mathematics.float3)"/>
    static
    float EvaluateNormalizedCloudHeight(in float3 positionPS, float _LowestCloudAltitude, float _HighestCloudAltitude)
    {
        return RangeRemap(_LowestCloudAltitude, _HighestCloudAltitude, length(positionPS));
    }

    /// <summary>Animation of the cloud shape position</summary>
    float3 AnimateShapeNoisePosition(float3 positionPS)
    {
        return AnimateShapeNoisePosition(positionPS, _WindVector.xy, _MediumWindSpeed, _VerticalShapeWindDisplacement);
    }

    /// <inheritdoc cref="AnimateShapeNoisePosition(Unity.Mathematics.float3)"/>
    static
    float3 AnimateShapeNoisePosition(in float3 positionPSTemp, in float2 _WindVector, float _MediumWindSpeed, float _VerticalShapeWindDisplacement)
    {
        float3 positionPS = positionPSTemp;

        // We reduce the top-view repetition of the pattern
        positionPS.y += (positionPS.x / 3.0f + positionPS.z / 7.0f);
        // We add the contribution of the wind displacements
        return positionPS + float3(_WindVector.x, 0.0f, _WindVector.y) * _MediumWindSpeed + float3(0.0f, _VerticalShapeWindDisplacement, 0.0f);
        //return positionPS;
    }

    /// <summary>Animation of the cloud erosion position</summary>
    float3 AnimateErosionNoisePosition(float3 positionPS)
    {
        return AnimateErosionNoisePosition(positionPS, _WindVector.xy, _SmallWindSpeed, _VerticalErosionWindDisplacement);
    }

    /// <inheritdoc cref="AnimateErosionNoisePosition(Unity.Mathematics.float3)"/>
    static
    float3 AnimateErosionNoisePosition(in float3 positionPS, in float2 _WindVector, float _SmallWindSpeed, float _VerticalErosionWindDisplacement)
    {
        return positionPS + float3(_WindVector.x, 0.0f, _WindVector.y) * _SmallWindSpeed + float3(0.0f, _VerticalErosionWindDisplacement, 0.0f);
        //return positionPS;
    }

    /// <summary>Structure that holds all the data used to define the cloud density of a point in space</summary>
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct CloudCoverageData
    {
        /// <summary>From a top-down view, in what proportions this pixel has clouds</summary>
        public
        float coverage;
        /// <summary>From a top-down view, in what proportions this pixel has clouds</summary>
        public
        float rainClouds;
        /// <summary>Value that allows us to request the cloudtype using the density</summary>
        public
        float cloudType;
        /// <summary>Maximal cloud height</summary>
        public
        float maxCloudHeight;
    };

    /// <summary>Function that evaluates the coverage data for a given point in planet space</summary>
    static
    void GetCloudCoverageData(float3 positionPS, out CloudCoverageData data)
    {
        // Convert the position into dome space and center the texture is centered above (0, 0, 0)
        //float2 normalizedPosition = AnimateCloudMapPosition(positionPS).xz / _NormalizationFactor * _CloudMapTiling.xy + _CloudMapTiling.zw - 0.5;
        //#if defined(CLOUDS_SIMPLE_PRESET)
        float4 cloudMapData = float4(0.9f, 0.0f, 0.25f, 1.0f);
        //#else
        //float4 cloudMapData = SAMPLE_TEXTURE2D_LOD(_CloudMapTexture, s_linear_repeat_sampler, float2(normalizedPosition), 0);
        //#endif
        data.coverage = cloudMapData.x;
        data.rainClouds = cloudMapData.y;
        data.cloudType = cloudMapData.z;
        data.maxCloudHeight = cloudMapData.w;
    }

    /// <summary>Density remapping function</summary>
    static
    float DensityRemap(float x, float a, float b, float c, float d)
    {
        return (((x - a) * rcp(b - a)) * (d - c)) + c;
    }

#if UNUSED
    // Horizon zero dawn technique to darken the clouds
    float PowderEffect(half cloudDensity, half cosAngle, half intensity)
    {
        float powderEffect = 1.0f - exp(-cloudDensity * 4.0f);
        powderEffect = saturate(powderEffect * 2.0f);
        return lerp(1.0f, lerp(1.0f, powderEffect, smoothstep(0.5f, -0.5f, cosAngle)), intensity);
    }
#endif // UNUSED

    /// <summary>Function that evaluates the cloud properties at a given absolute world space position</summary>
    void EvaluateCloudProperties(float3 positionPS, float noiseMipOffset, float erosionMipOffset, bool cheapVersion, bool lightSampling,
                                out CloudProperties properties)
    {
        // Initialize all the values to 0 in case
        properties = default;

        //#ifndef CLOUDS_SIMPLE_PRESET
        // When using a cloud map, we cannot support the full planet due to UV issues
        //#endif

        // Remove global clouds below the horizon
        if (!_LOCAL_VOLUMETRIC_CLOUDS
            && positionPS.y < _EarthRadius)
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
        float3 baseNoiseSamplingCoordinates = float3(AnimateShapeNoisePosition(positionPS).xzy / NOISE_TEXTURE_NORMALIZATION_FACTOR) * _ShapeScale - float3(_ShapeNoiseOffset.x, _ShapeNoiseOffset.y, _VerticalShapeNoiseOffset);

        // Evaluate the coordinates at which the noise will be sampled and apply wind displacement
        baseNoiseSamplingCoordinates += _AltitudeDistortion * properties.height * float3(_WindDirection.x, _WindDirection.y, 0.0f);

        // Read the low frequency Perlin-Worley and Worley noises
        float lowFrequencyNoise = SAMPLE_TEXTURE3D_LOD(_Worley128RGBA, s_trilinear_repeat_sampler, baseNoiseSamplingCoordinates, noiseMipOffset);

        // Evaluate the cloud coverage data for this position
        CloudCoverageData cloudCoverageData;
        GetCloudCoverageData(positionPS, out cloudCoverageData);

        // If this region of space has no cloud coverage, exit right away
        if (cloudCoverageData.coverage <= CLOUD_DENSITY_TRESHOLD || cloudCoverageData.maxCloudHeight < properties.height)
            return;

        // Read from the LUT
        //#if defined(CLOUDS_SIMPLE_PRESET)
        float3 densityErosionAO = SAMPLE_TEXTURE2D_LOD(_CloudCurveTexture, s_linear_repeat_sampler, float2(0.0f, properties.height), 0).xyz;
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
            float3 erosionCoords = AnimateErosionNoisePosition(positionPS) / (NOISE_TEXTURE_NORMALIZATION_FACTOR * _ErosionScale);
            float erosionNoise = 1.0f - SAMPLE_TEXTURE3D_LOD(_ErosionNoise, s_linear_repeat_sampler, erosionCoords, CLOUD_DETAIL_MIP_OFFSET + erosionMipOffset);
            erosionNoise = lerp(0.0f, erosionNoise, erosionFactor * 0.75f * cloudCoverageData.coverage);
            properties.ambientOcclusion = saturate(properties.ambientOcclusion - sqrt(erosionNoise * _ErosionOcclusion));
            base_cloud = DensityRemap(base_cloud, erosionNoise, 1.0f, 0.0f, 1.0f);

            if (_CLOUDS_MICRO_EROSION)
            {
                float3 fineCoords = AnimateErosionNoisePosition(positionPS) / (NOISE_TEXTURE_NORMALIZATION_FACTOR * _MicroErosionScale);
                float fineNoise = 1.0f - SAMPLE_TEXTURE3D_LOD(_ErosionNoise, s_linear_repeat_sampler, fineCoords, CLOUD_DETAIL_MIP_OFFSET + erosionMipOffset);
                fineNoise = lerp(0.0f, fineNoise, microDetailFactor * 0.5f * cloudCoverageData.coverage);
                base_cloud = DensityRemap(base_cloud, fineNoise, 1.0f, 0.0f, 1.0f);
            }
        }

        // Make sure we do not send any negative values
        base_cloud = max(0.0f, base_cloud);

        // Attenuate everything by the density multiplier
        properties.density = base_cloud * _DensityMultiplier;
    }

    VolumetricRayResult TraceVolumetricRay(in CloudRay cloudRay)
    {
        VolumetricRayResult volumetricRay;
        volumetricRay.scattering = default;
        volumetricRay.transmittance = 1.0f;
        volumetricRay.meanDistance = FLT_MAX;
        volumetricRay.invalidRay = true;

        // Determine if ray intersects bounding volume, if the ray does not intersect the cloud volume AABB, skip right away
        RayMarchRange rayMarchRange;
        if (GetCloudVolumeIntersection(cloudRay.originWS, half3(cloudRay.direction), out rayMarchRange))
        {
            if (cloudRay.maxRayLength >= rayMarchRange.start)
            {
                // Initialize the depth for accumulation
                volumetricRay.meanDistance = 0.0f;

                // Total distance that the ray must travel including empty spaces
                // Clamp the travel distance to whatever is closer
                // - Sky Occluder
                // - Volume end
                // - Far plane
                float totalDistance = min(rayMarchRange.end, cloudRay.maxRayLength) - rayMarchRange.start;

                // Evaluate our integration step
                float stepS = min(totalDistance / (float)_NumPrimarySteps, _MaxStepSize);
                totalDistance = stepS * _NumPrimarySteps;

                // Compute the environment lighting that is going to be used for the cloud evaluation
                /*
                float3 rayMarchStartPS = ConvertToPS(cloudRay.originWS) + rayMarchRange.start * cloudRay.direction;
                float3 rayMarchEndPS = rayMarchStartPS + totalDistance * cloudRay.direction;
                */

                // Tracking the number of steps that have been made
                int currentIndex = 0;

                // Normalization value of the depth
                float meanDistanceDivider = 0.0f;

                // Current position for the evaluation, apply blue noise to start position
                float currentDistance = cloudRay.integrationNoise;
                float3 currentPositionWS = cloudRay.originWS + (rayMarchRange.start + currentDistance) * cloudRay.direction;

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

                    // Accumulate in WS and convert at each iteration to avoid precision issues
                    float3 currentPositionPS = ConvertToPS(currentPositionWS);

                    // Should we be evaluating the clouds or just doing the large ray marching
                    if (activeSampling)
                    {
                        // If the density is null, we can skip as there will be no contribution
                        CloudProperties properties;
                        EvaluateCloudProperties(currentPositionPS, 0.0f, erosionMipOffset, false, false, out properties);

                        // Apply the fade in function to the density
                        properties.density *= densityAttenuationValue;

                        if (properties.density > CLOUD_DENSITY_TRESHOLD)
                        {
                            // Contribute to the average depth (must be done first in case we end up inside a cloud at the next step)
                            float transmitanceXdensity = volumetricRay.transmittance * properties.density;
                            volumetricRay.meanDistance += (rayMarchRange.start + currentDistance) * transmitanceXdensity;
                            meanDistanceDivider += transmitanceXdensity;

                            // Evaluate the cloud at the position
                            //EvaluateCloud(properties, cloudRay.direction, currentPositionWS, rayMarchStartPS, rayMarchEndPS, stepS, currentDistance / totalDistance, volumetricRay);
                            // No lighting Version
                            {
                                float extinction = properties.density * properties.sigmaT;
                                float transmittance = exp(-extinction * stepS);
                                volumetricRay.transmittance *= transmittance;
                            }

                            // if most of the energy is absorbed, just leave.
                            if (volumetricRay.transmittance < 0.003f)
                            {
                                volumetricRay.transmittance = 0.0f;
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
                        float relativeStepSize = lerp(cloudRay.integrationNoise, 1.0f, saturate(currentIndex));
                        currentPositionWS += stepS * relativeStepSize * cloudRay.direction;
                        currentDistance += stepS * relativeStepSize;

                    }
                    else
                    {
                        CloudProperties properties;
                        EvaluateCloudProperties(currentPositionPS, 1.0f, 0.0f, true, false, out properties);

                        // Apply the fade in function to the density
                        properties.density *= densityAttenuationValue;

                        // If the density is lower than our tolerance,
                        if (properties.density < CLOUD_DENSITY_TRESHOLD)
                        {
                            currentPositionWS += stepS * 2.0f * cloudRay.direction;
                            currentDistance += stepS * 2.0f;
                        }
                        else
                        {
                            // Somewhere between this step and the previous clouds started
                            // We reset all the counters and enable active sampling
                            currentPositionWS -= cloudRay.direction * stepS;
                            currentDistance -= stepS;
                            currentIndex -= 1;
                            activeSampling = true;
                            sequentialEmptySamples = 0;
                        }
                    }

                    currentIndex++;
                }

                // Normalized the depth we computed
                if (volumetricRay.meanDistance != 0.0f)
                {
                    volumetricRay.invalidRay = false;
                    volumetricRay.meanDistance /= meanDistanceDivider;
                }
                else
                {
                    volumetricRay.invalidRay = true;
                }
            }
        }

        return volumetricRay;
    }

    public
    void UpdateCloudsProperties()
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
#if _CONST_EARTH_RADIUS
#else
        _EarthRadius = cloudsMaterial.GetFloat(earthRadius);
#endif // _CONST_EARTH_RADIUS
        _PlanetCenterPosition = new float3(0.0f, -_EarthRadius, 0.0f);
    }

    /// <summary>
    /// Calculates the density of volumetric clouds on the CPU along a ray from a given start position in world space.
    /// </summary>
    /// <param name="startPosWS">The start position of the ray in world space.</param>
    /// <param name="directionWS">The normalized direction of the ray in world space.</param>
    /// <returns>
    /// The cloud density along the ray.
    /// </returns>
    public float QueryCloudsRay(float3 startPosWS, float3 directionWS)
    {
        if (cloudsMaterial == null)
            return 0.0f;

        UpdateCloudsProperties();

        CloudRay ray;
        ray.originWS = startPosWS;
        ray.direction = directionWS;
        ray.maxRayLength = MAX_SKYBOX_VOLUMETRIC_CLOUDS_DISTANCE;
        ray.integrationNoise = 0.0f;

        VolumetricRayResult volumetricRay = TraceVolumetricRay(in ray);

        return volumetricRay.invalidRay ? 0.0f : 1.0f - volumetricRay.transmittance;
    }

    /*
    // An example of query clouds density
    private void Update()
    {
        float3 startPosWS = new float3(0.0f, 0.0f, 0.0f);
        float3 directionWS = new float3(0.0f, 1.0f, 0.0f); // make sure it's normalized
        float density = QueryCloudsRay(startPosWS , directionWS);

        Debug.Log(density);
    }
    */

    #region Burst
    /// <inheritdoc cref="QueryCloudsRay(Unity.Mathematics.float3, Unity.Mathematics.float3)"/>
    /// <param name="results">The results array must have been created with at least
    /// as many elements as the input NativeArrays. It cannot use <see cref="Unity.Collections.Allocator.Temp"/>.</param>
    public JobHandle QueryCloudsRay(float3 startPosWS, float3 directionWS, NativeArray<float> results,
        JobHandle dependency = default)
    {
        var startPosWSArray = new NativeArray<float3>(1, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var directionWSArray = new NativeArray<float3>(1, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        startPosWSArray[0] = startPosWS;
        directionWSArray[0] = directionWS;

        return QueryCloudsRay(startPosWSArray, directionWSArray, results, dependency);
    }

    /// <inheritdoc cref="QueryCloudsRay(Unity.Mathematics.float3, Unity.Mathematics.float3, Unity.Collections.NativeArray{float}, Unity.Jobs.JobHandle)"/>
    /// <remarks><paramref name="startPosWS"/> and <paramref name="directionWS"/> will be automatically disposed.</remarks>
    public JobHandle QueryCloudsRay(NativeArray<float3> startPosWS, NativeArray<float3> directionWS, NativeArray<float> results,
        JobHandle dependency = default)
    {
        UnityEngine.Assertions.Assert.IsTrue(startPosWS.IsCreated);
        UnityEngine.Assertions.Assert.IsTrue(directionWS.IsCreated);
        UnityEngine.Assertions.Assert.IsTrue(results.IsCreated);

        UnityEngine.Assertions.Assert.AreEqual(startPosWS.Length, directionWS.Length);
        UnityEngine.Assertions.Assert.IsTrue(results.Length >= startPosWS.Length, "results must contain at least as many elements as the input NativeArrays.");

        var cloudRay = new NativeArray<CloudRay>(startPosWS.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        var cloudRayHandle = new CloudRayJob
        {
            startPosWS = startPosWS,
            directionWS = directionWS,
            cloudRay = cloudRay,
        };

        dependency = cloudRayHandle.Schedule(cloudRay.Length, dependency);

        var traceVolumetricRayHandle = new TraceVolumetricRayJob
        {
            cloudRay = cloudRay,

            _PlanetCenterPosition = _PlanetCenterPosition,
            _AltitudeDistortion = _AltitudeDistortion,

            _ShapeNoiseOffset = _ShapeNoiseOffset.xy,
            _WindDirection = _WindDirection.xy,
            _WindVector = _WindVector.xy,

            _DensityMultiplier = _DensityMultiplier,
            _ErosionFactor = _ErosionFactor,
            _ErosionOcclusion = _ErosionOcclusion,
            _ErosionScale = _ErosionScale,
            _FadeInDistance = _FadeInDistance,
            _FadeInStart = _FadeInStart,
            _HighestCloudAltitude = _HighestCloudAltitude,
            _LowestCloudAltitude = _LowestCloudAltitude,
            _MaxStepSize = _MaxStepSize,
            _MediumWindSpeed = _MediumWindSpeed,
            _MicroErosionFactor = _MicroErosionFactor,
            _MicroErosionScale = _MicroErosionScale,
            _NumPrimarySteps = _NumPrimarySteps,
            _ShapeFactor = _ShapeFactor,
            _ShapeScale = _ShapeScale,
            _SmallWindSpeed = _SmallWindSpeed,
            _VerticalErosionWindDisplacement = _VerticalErosionWindDisplacement,
            _VerticalShapeNoiseOffset = _VerticalShapeNoiseOffset,
            _VerticalShapeWindDisplacement = _VerticalShapeWindDisplacement,

            _CLOUDS_MICRO_EROSION = _CLOUDS_MICRO_EROSION,
            _LOCAL_VOLUMETRIC_CLOUDS = _LOCAL_VOLUMETRIC_CLOUDS,

            _Worley128RGBA = _Worley128RGBA.GetPixelData<byte>(mipLevel: 0),
            _Worley128RGBAWidth = _Worley128RGBA.width,
            _Worley128RGBAHeight = _Worley128RGBA.height,

            _ErosionNoise = _ErosionNoise.GetPixelData<byte>(mipLevel: 0),
            _ErosionNoiseWidth = _ErosionNoise.width,
            _ErosionNoiseHeight = _ErosionNoise.height,

            _CloudCurveTexture = _CloudCurveTexture.GetPixelData<half4>(mipLevel: 0),
            _CloudCurveTextureWidth = _CloudCurveTexture.width,

            results = results,
        };

        return traceVolumetricRayHandle.ScheduleByRef(cloudRay.Length, dependency);
    }

#if ENABLE_BURST_1_0_0_OR_NEWER
    [BurstCompile(FloatMode = FloatMode.Fast)]
#endif // ENABLE_BURST_1_0_0_OR_NEWER
    struct CloudRayJob : IJobFor
    {
        [ReadOnly][NativeMatchesParallelForLength][DeallocateOnJobCompletion] public NativeArray<float3> startPosWS;        
        [ReadOnly][NativeMatchesParallelForLength][DeallocateOnJobCompletion] public NativeArray<float3> directionWS;
        [WriteOnly][NativeMatchesParallelForLength] public NativeArray<CloudRay> cloudRay;

        public void Execute(int index)
        {
#if ZERO
            UnityEngine.Assertions.Assert.AreApproximatelyEqual(1f, ((Vector3)directionWS[index]).magnitude, "directionWS must be normalised.");
#endif // ZERO

            cloudRay[index] = new CloudRay
            {
                originWS = startPosWS[index],
                direction = directionWS[index],
                maxRayLength = MAX_SKYBOX_VOLUMETRIC_CLOUDS_DISTANCE,
                integrationNoise = default,
            };
        }
    }

#if ENABLE_BURST_1_0_0_OR_NEWER
    [BurstCompile(FloatMode = FloatMode.Fast)]
#endif // ENABLE_BURST_1_0_0_OR_NEWER
    struct TraceVolumetricRayJob : IJobFor
    {
        [NativeMatchesParallelForLength]
        [DeallocateOnJobCompletion]
        [ReadOnly] public NativeArray<CloudRay> cloudRay;

        [ReadOnly] public float3 _PlanetCenterPosition;
        [ReadOnly] public float _AltitudeDistortion;

        [ReadOnly] public float2 _ShapeNoiseOffset;
        [ReadOnly] public float2 _WindDirection;
        [ReadOnly] public float2 _WindVector;

        [ReadOnly] public float _DensityMultiplier;
        [ReadOnly] public float _ErosionFactor;
        [ReadOnly] public float _ErosionOcclusion;
        [ReadOnly] public float _ErosionScale;
        [ReadOnly] public float _FadeInDistance;
        [ReadOnly] public float _FadeInStart;
        [ReadOnly] public float _HighestCloudAltitude;
        [ReadOnly] public float _LowestCloudAltitude;
        [ReadOnly] public float _MaxStepSize;
        [ReadOnly] public float _MediumWindSpeed;
        [ReadOnly] public float _MicroErosionFactor;
        [ReadOnly] public float _MicroErosionScale;
        [ReadOnly] public float _NumPrimarySteps;
        [ReadOnly] public float _ShapeFactor;
        [ReadOnly] public float _ShapeScale;
        [ReadOnly] public float _SmallWindSpeed;
        [ReadOnly] public float _VerticalErosionWindDisplacement;
        [ReadOnly] public float _VerticalShapeNoiseOffset;
        [ReadOnly] public float _VerticalShapeWindDisplacement;

        [ReadOnly] public bool _CLOUDS_MICRO_EROSION;
        [ReadOnly] public bool _LOCAL_VOLUMETRIC_CLOUDS;

        [NativeFixedLength(128 * 128 * 128)]
        [ReadOnly] public NativeArray<byte> _Worley128RGBA;
        [ReadOnly] public int _Worley128RGBAWidth;
        [ReadOnly] public int _Worley128RGBAHeight;

        [NativeFixedLength(32 * 32 * 32)]
        [ReadOnly] public NativeArray<byte> _ErosionNoise;
        [ReadOnly] public int _ErosionNoiseWidth;
        [ReadOnly] public int _ErosionNoiseHeight;

        // Allow _CloudCurveTexture to be accessed from a background thread
        // while PrepareCustomLutData() is writing to it
        [Unity.Collections.LowLevel.Unsafe.NativeDisableContainerSafetyRestriction]
        [ReadOnly] public NativeArray<half4> _CloudCurveTexture;
        [ReadOnly] public int _CloudCurveTextureWidth;

        [NativeMatchesParallelForLength]
        [WriteOnly] public NativeArray<float> results;

        public void Execute(int index)
        {
            CloudRay ray = cloudRay[index];
            TraceVolumetricRay(in ray, out var volumetricRay);

            results[index] = volumetricRay.invalidRay ? 0.0f : 1.0f - volumetricRay.transmittance;
        }

        #region readonly
        readonly void TraceVolumetricRay(in CloudRay cloudRay, out VolumetricRayResult volumetricRay)
        {
            volumetricRay.scattering = default;
            volumetricRay.transmittance = 1.0f;
            volumetricRay.meanDistance = FLT_MAX;
            volumetricRay.invalidRay = true;

            // Determine if ray intersects bounding volume, if the ray does not intersect the cloud volume AABB, skip right away
            RayMarchRange rayMarchRange;
            if (!IntersectCloudVolume(ConvertToPS(cloudRay.originWS, _PlanetCenterPosition), half3(cloudRay.direction),
                _LowestCloudAltitude, _HighestCloudAltitude, out rayMarchRange.start, out rayMarchRange.end)
                || cloudRay.maxRayLength < rayMarchRange.start)
            {
                return;
            }

            // Initialize the depth for accumulation
            volumetricRay.meanDistance = 0.0f;

            // Total distance that the ray must travel including empty spaces
            // Clamp the travel distance to whatever is closer
            // - Sky Occluder
            // - Volume end
            // - Far plane
            float totalDistance = min(rayMarchRange.end, cloudRay.maxRayLength) - rayMarchRange.start;

            // Evaluate our integration step
            float stepS = min(totalDistance / _NumPrimarySteps, _MaxStepSize);
            totalDistance = stepS * _NumPrimarySteps;

            // Compute the environment lighting that is going to be used for the cloud evaluation
            /*
            float3 rayMarchStartPS = ConvertToPS(cloudRay.originWS) + rayMarchRange.start * cloudRay.direction;
            float3 rayMarchEndPS = rayMarchStartPS + totalDistance * cloudRay.direction;
            */

            // Tracking the number of steps that have been made
            int currentIndex = 0;

            // Normalization value of the depth
            float meanDistanceDivider = 0.0f;

            // Current position for the evaluation, apply blue noise to start position
            float currentDistance = cloudRay.integrationNoise;
            float3 currentPositionWS = cloudRay.originWS + (rayMarchRange.start + currentDistance) * cloudRay.direction;

            // Initialize the values for the optimized ray marching
            bool activeSampling = true;
            int sequentialEmptySamples = 0;

            // Do the ray march for every step that we can.
            while (currentIndex < (int)_NumPrimarySteps && currentDistance < totalDistance)
            {
                // Compute the camera-distance based attenuation
                float densityAttenuationValue = DensityFadeValue(rayMarchRange.start + currentDistance, _FadeInStart, _FadeInDistance);
                /*
                // Compute the mip offset for the erosion texture
                float erosionMipOffset = ErosionMipOffset(rayMarchRange.start + currentDistance);
                */

                // Accumulate in WS and convert at each iteration to avoid precision issues
                float3 currentPositionPS = ConvertToPS(currentPositionWS, _PlanetCenterPosition);

                // Should we be evaluating the clouds or just doing the large ray marching
                if (activeSampling)
                {
                    // If the density is null, we can skip as there will be no contribution
                    CloudProperties properties;
                    EvaluateCloudProperties(currentPositionPS, cheapVersion: false, out properties);

                    // Apply the fade in function to the density
                    properties.density *= densityAttenuationValue;

                    if (properties.density > CLOUD_DENSITY_TRESHOLD)
                    {
                        // Contribute to the average depth (must be done first in case we end up inside a cloud at the next step)
                        float transmitanceXdensity = volumetricRay.transmittance * properties.density;
                        volumetricRay.meanDistance += (rayMarchRange.start + currentDistance) * transmitanceXdensity;
                        meanDistanceDivider += transmitanceXdensity;

                        // Evaluate the cloud at the position
                        //EvaluateCloud(properties, cloudRay.direction, currentPositionWS, rayMarchStartPS, rayMarchEndPS, stepS, currentDistance / totalDistance, volumetricRay);
                        // No lighting Version
                        {
                            float extinction = properties.density * properties.sigmaT;
                            float transmittance = exp(-extinction * stepS);
                            volumetricRay.transmittance *= transmittance;
                        }

                        // if most of the energy is absorbed, just leave.
                        if (volumetricRay.transmittance < 0.003f)
                        {
                            volumetricRay.transmittance = 0.0f;
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
                    float relativeStepSize = lerp(cloudRay.integrationNoise, 1.0f, saturate(currentIndex));
                    currentPositionWS += stepS * relativeStepSize * cloudRay.direction;
                    currentDistance += stepS * relativeStepSize;

                }
                else
                {
                    CloudProperties properties;
                    EvaluateCloudProperties(currentPositionPS, cheapVersion: true, out properties);

                    // Apply the fade in function to the density
                    properties.density *= densityAttenuationValue;

                    // If the density is lower than our tolerance,
                    if (properties.density < CLOUD_DENSITY_TRESHOLD)
                    {
                        currentPositionWS += stepS * 2.0f * cloudRay.direction;
                        currentDistance += stepS * 2.0f;
                    }
                    else
                    {
                        // Somewhere between this step and the previous clouds started
                        // We reset all the counters and enable active sampling
                        currentPositionWS -= cloudRay.direction * stepS;
                        currentDistance -= stepS;
                        currentIndex -= 1;
                        activeSampling = true;
                        sequentialEmptySamples = 0;
                    }
                }

                currentIndex++;
            }

            // Normalized the depth we computed
            if (volumetricRay.meanDistance != 0.0f)
            {
                volumetricRay.invalidRay = false;
                volumetricRay.meanDistance /= meanDistanceDivider;
            }
            else
            {
                volumetricRay.invalidRay = true;
            }
        }

        readonly void EvaluateCloudProperties(in float3 positionPS, bool cheapVersion,
            out CloudProperties properties)
        {
            // Initialize all the values to 0 in case
            properties = default;

            //#ifndef CLOUDS_SIMPLE_PRESET
            // When using a cloud map, we cannot support the full planet due to UV issues
            //#endif

            // Remove global clouds below the horizon
            if (!_LOCAL_VOLUMETRIC_CLOUDS
                && positionPS.y < _EarthRadius)
                return;

            // By default the ambient occlusion is 1.0
            properties.ambientOcclusion = 1.0f;

            // Evaluate the normalized height of the position within the cloud volume
            properties.height = EvaluateNormalizedCloudHeight(positionPS, _LowestCloudAltitude, _HighestCloudAltitude);

            // When rendering in camera space, we still want horizontal scrolling
            /*
            if (!_LOCAL_VOLUMETRIC_CLOUDS)
            {
                positionPS.x += _WorldSpaceCameraPos.x;
                positionPS.z += _WorldSpaceCameraPos.z;
            }
            */

            // Evaluate the generic sampling coordinates
            float3 baseNoiseSamplingCoordinates = float3(AnimateShapeNoisePosition(positionPS, _WindVector, _MediumWindSpeed, _VerticalShapeWindDisplacement).xzy / NOISE_TEXTURE_NORMALIZATION_FACTOR) * _ShapeScale - float3(_ShapeNoiseOffset.x, _ShapeNoiseOffset.y, _VerticalShapeNoiseOffset);

            // Evaluate the coordinates at which the noise will be sampled and apply wind displacement
            baseNoiseSamplingCoordinates += _AltitudeDistortion * properties.height * float3(_WindDirection.x, _WindDirection.y, 0.0f);

            // Read the low frequency Perlin-Worley and Worley noises
            float lowFrequencyNoise = SAMPLE_TEXTURE3D_LOD(_Worley128RGBA, baseNoiseSamplingCoordinates, _Worley128RGBAWidth, _Worley128RGBAHeight);

            // Evaluate the cloud coverage data for this position
            CloudCoverageData cloudCoverageData;
            GetCloudCoverageData(positionPS, out cloudCoverageData);

            // If this region of space has no cloud coverage, exit right away
            if (cloudCoverageData.coverage <= CLOUD_DENSITY_TRESHOLD || cloudCoverageData.maxCloudHeight < properties.height)
                return;

            // Read from the LUT
            //#if defined(CLOUDS_SIMPLE_PRESET)
            float3 densityErosionAO = SAMPLE_TEXTURE2D_LOD(_CloudCurveTexture, properties.height, _CloudCurveTextureWidth);
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
                float3 erosionCoords = AnimateErosionNoisePosition(positionPS, _WindVector, _SmallWindSpeed, _VerticalErosionWindDisplacement) / (NOISE_TEXTURE_NORMALIZATION_FACTOR * _ErosionScale);
                float erosionNoise = 1.0f - SAMPLE_TEXTURE3D_LOD(_ErosionNoise, erosionCoords, _ErosionNoiseWidth, _ErosionNoiseHeight);
                erosionNoise = lerp(0.0f, erosionNoise, erosionFactor * 0.75f * cloudCoverageData.coverage);
                properties.ambientOcclusion = saturate(properties.ambientOcclusion - sqrt(erosionNoise * _ErosionOcclusion));
                base_cloud = DensityRemap(base_cloud, erosionNoise, 1.0f, 0.0f, 1.0f);

                if (_CLOUDS_MICRO_EROSION)
                {
                    float3 fineCoords = AnimateErosionNoisePosition(positionPS, _WindVector, _SmallWindSpeed, _VerticalErosionWindDisplacement) / (NOISE_TEXTURE_NORMALIZATION_FACTOR * _MicroErosionScale);
                    float fineNoise = 1.0f - SAMPLE_TEXTURE3D_LOD(_ErosionNoise, fineCoords, _ErosionNoiseWidth, _ErosionNoiseHeight);
                    fineNoise = lerp(0.0f, fineNoise, microDetailFactor * 0.5f * cloudCoverageData.coverage);
                    base_cloud = DensityRemap(base_cloud, fineNoise, 1.0f, 0.0f, 1.0f);
                }
            }

            // Make sure we do not send any negative values
            base_cloud = max(0.0f, base_cloud);

            // Attenuate everything by the density multiplier
            properties.density = base_cloud * _DensityMultiplier;
        }
        #endregion // readonly

        #region static
        [System.Runtime.CompilerServices.MethodImpl(256)]
        static float3 ConvertToPS(in float3 x, in float3 _PlanetCenterPosition) => x - _PlanetCenterPosition;

        [System.Runtime.CompilerServices.MethodImpl(256)]
        static float SAMPLE_TEXTURE3D_LOD(in NativeArray<byte> tex3D, in float3 texCoord,
            [AssumeRange(1, int.MaxValue)] int width,
            [AssumeRange(1, int.MaxValue)] int height)
        {
            int depth = tex3D.Length / (width * height);
            byte r = tex3D[
                  mod((int)(texCoord.z * depth), depth) * width * height
                + mod((int)(texCoord.y * height), height) * width
                + mod((int)(texCoord.x * width), width)];

            const float normaliseByte = 1f / byte.MaxValue;
            return r * normaliseByte;
        }

        [System.Runtime.CompilerServices.MethodImpl(256)]
        static float3 SAMPLE_TEXTURE2D_LOD(in NativeArray<half4> tex2D, float texCoordy,
            [AssumeRange(1, int.MaxValue)] int width)
        {
            int height = tex2D.Length / width;
            half4 c = tex2D[mod((int)(texCoordy * height), height) * width];
            return c.xyz;
        }

#pragma warning disable IDE1006 // Naming Styles
        /// <see href="https://stackoverflow.com/a/74552262"/>
        [System.Runtime.CompilerServices.MethodImpl(256)]
        [return: AssumeRange(0, int.MaxValue)]
        static int mod(int a, [AssumeRange(1, int.MaxValue)] int b)
        {
            return (a % b + b) % b;
        }
#pragma warning restore IDE1006 // Naming Styles
        #endregion // static
    }
    #endregion // Burst
}
