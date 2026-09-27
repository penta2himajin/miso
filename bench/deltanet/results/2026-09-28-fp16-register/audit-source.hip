#include "live.hpp"
#include "register.hpp"
#include "volatile.hpp"
extern "C" __global__ void __launch_bounds__(256)
    volatile_step(miso::bench::audit_volatile::StepParams p) {
  miso::bench::audit_volatile::step_op<miso::bench::audit_volatile::Output::Fp16>(p, 0, 32);
}
extern "C" __global__ void __launch_bounds__(256) live_step(miso::bench::audit_live::StepParams p) {
  miso::bench::audit_live::step_op<miso::bench::audit_live::Output::Fp16>(p, 0, 32);
}
extern "C" __global__ void __launch_bounds__(256)
    register_step(miso::bench::audit_register::StepParams p) {
  miso::bench::audit_register::step_op<miso::bench::audit_register::Output::Fp16>(p, 0, 32);
}
extern "C" __global__ void __launch_bounds__(256)
    baseline_step(miso::kernels::DeltaNetStepParams p) {
  miso::kernels::deltanet_step_op<256>(p, 0, 32);
}
