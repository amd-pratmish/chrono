// =============================================================================
// Settling-bed performance harness for GitHub issue #807 (AMD Instinct vs CUDA).
// Usage:
//   demo_DEM_settling_bench [--grains N] [--steps S] [--radius R] [--box L]
// =============================================================================

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "chrono/core/ChVector3.h"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_dem/physics/ChSystemDem.h"

using namespace chrono;
using namespace chrono::dem;

static unsigned int parse_u32(int argc, char** argv, const char* flag, unsigned int default_val) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return static_cast<unsigned int>(std::stoul(argv[i + 1]));
        }
    }
    return default_val;
}

static float parse_f32(int argc, char** argv, const char* flag, float default_val) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return std::stof(argv[i + 1]);
        }
    }
    return default_val;
}

int main(int argc, char* argv[]) {
    const unsigned int target_grains = parse_u32(argc, argv, "--grains", 159600);
    const unsigned int target_steps = parse_u32(argc, argv, "--steps", 25000);
    const float sphere_radius = parse_f32(argc, argv, "--radius", 0.005f);
    const float box_len = parse_f32(argc, argv, "--box", 1.0f);

    const float density = 2600.f;
    const float step = 5e-6f;

    ChSystemDem dem(sphere_radius, density, ChVector3f(box_len, box_len, box_len));

    dem.SetGravitationalAcceleration(ChVector3f(0, 0, -9.81f));
    dem.SetFixedStepSize(step);
    dem.SetFrictionMode(CHDEM_FRICTION_MODE::MULTI_STEP);
    dem.SetTimeIntegrator(CHDEM_TIME_INTEGRATOR::EXTENDED_TAYLOR);
    dem.SetYoungModulus_SPH(1e8);
    dem.SetYoungModulus_WALL(1e8);
    dem.SetPoissonRatio_SPH(0.3);
    dem.SetPoissonRatio_WALL(0.3);
    dem.SetRestitution_SPH(0.2);
    dem.SetRestitution_WALL(0.2);
    dem.SetStaticFrictionCoeff_SPH2SPH(0.5f);
    dem.SetStaticFrictionCoeff_SPH2WALL(0.5f);

    // Regular grid slightly below the box top (settling bed).
    const float spacing = 2.01f * sphere_radius;
    const int nx = static_cast<int>(std::cbrt(static_cast<double>(target_grains)));
    const int ny = nx;
    const int nz = nx;
    const float origin = -0.5f * box_len + 2.f * sphere_radius;

    std::vector<ChVector3f> points;
    points.reserve(static_cast<size_t>(nx) * ny * nz);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (points.size() >= target_grains) {
                    break;
                }
                points.emplace_back(origin + i * spacing, origin + j * spacing, origin + k * spacing);
            }
        }
    }

    std::cout << "grains=" << points.size() << " steps=" << target_steps << " step=" << step << " radius=" << sphere_radius
              << std::endl;
    std::cout.flush();

    dem.SetParticles(points);
    dem.Initialize();

    // Warmup (JIT/cache/broadphase); not included in reported time.
    const unsigned int warmup_steps = parse_u32(argc, argv, "--warmup", 500);
    if (warmup_steps > 0) {
        dem.AdvanceSimulation(static_cast<float>(step * warmup_steps));
    }

    const double advance_time = step * target_steps;
    const auto t0 = std::chrono::steady_clock::now();
    dem.AdvanceSimulation(static_cast<float>(advance_time));
    const auto t1 = std::chrono::steady_clock::now();

    const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    const double ns_per_grain_step = elapsed_s * 1e9 / (static_cast<double>(points.size()) * target_steps);

    std::cout << "elapsed_s=" << elapsed_s << std::endl;
    std::cout << "ns_per_grain_step=" << ns_per_grain_step << std::endl;

    return 0;
}
