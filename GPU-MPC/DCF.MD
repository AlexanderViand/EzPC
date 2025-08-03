  # DCF vs DDCF
  
  Note: In the following, greek letters are used for parameters that need to be known during offline setup phase,
  while latin letters indicate variables provided by the parties during the online phase.
  
  DCF (Distributed Comparison Function):
  - Defined as: f^<_{α,β}(x) = β if x < α, and 0 otherwise
  - Outputs a single value β for inputs less than α
  - Implementation uses a binary tree traversal with correction words

  DDCF (Dual Distributed Comparison Function):
  - Defined as: f_{α,β₁,β₂}(x) = β₁ if x < α, and β₂ if x ≥ α
  - Outputs two different values depending on the comparison
  - Key implementation insight from the paper (Section F.2): DDCF can be implemented using DCF

  The critical implementation relationship is:
  f_{α,β₁,β₂}(x) = β₂ + f^<_{α,β₁-β₂}(x)

  This means DDCF is essentially a DCF with an additive offset.

  ## Implementation Details

  From Figure 12 in the paper, the DDCF implementation:

  1. Key Generation (GenDDCF):
    - Set β = β₁ - β₂
    - Generate DCF keys for f^<_{α,β}
    - Sample random shares S₀, S₁ such that S₀ + S₁ = β₂
    - Each party's key contains: DCF key + share Sᵦ
  2. Evaluation (EvalDDCF):
    - Evaluate the DCF portion to get y^{(n-1)}_b
    - Add the share Sᵦ
    - Return y^{(n-1)}_b + Sᵦ

  The key size of DDCF is only ⌈log |G|⌉ bits larger than DCF.

## Integer Comparison

For secure integer comparison where parties P₀ and P₁ have private inputs x and y respectively (unknown during offline setup):

1. **Offline Phase**:
   - Choose random α ∈ Z_{2^n} and set β = 1
   - Generate DCF keys for f^<_{α,β}
   - P₀ receives k₀, P₁ receives k₁
   - Generate random mask shares r₀, r₁ such that r₀ + r₁ = α (mod 2^n)

2. **Online Phase**:
   - P₀ computes z₀ = x - r₀ (mod 2^n)
   - P₁ computes z₁ = y - r₁ (mod 2^n)
   - Parties exchange and reconstruct z = z₀ + z₁ = x - y + α (mod 2^n)
   - Each party evaluates: [[x < y]]_b = EvalDCF(k_b, z)

**Correctness**: Since z = x - y + α (mod 2^n):
- If x < y, then z < α (with high probability for random α)
- If x ≥ y, then z ≥ α (with high probability for random α)
- The DCF evaluation returns shares of 1 if z < α (i.e., x < y), and shares of 0 otherwise

**Security**: The protocol is secure because:
- z is uniformly random due to the mask α
- Neither party learns anything about x or y beyond the comparison result
- The DCF evaluation maintains the secret-sharing property

## DDCF-based Integer Comparison

The SCMP (Secure Comparison) implementation uses a more efficient protocol based on DDCF:

For secure integer comparison where parties P₀ and P₁ have shares of x and y respectively:

1. **Offline Phase**:
   - Choose random masks ρ₁, ρ₂ ∈ Z_{2^n}
   - Compute μ = -(ρ₁ - ρ₂) (mod 2^n)
   - Extract MSB: μ_msb = (μ >> (n-1)) & 1
   - Compute index: μ_idx = μ - (μ_msb << (n-1))
   - Set DDCF payloads: β₁ = 1 ⊕ μ_msb, β₂ = μ_msb
   - Generate DDCF keys for f_{μ_idx,β₁,β₂} with domain size 2^(n-1)
   - Generate random output mask shares r₀, r₁

2. **Online Phase**:
   - Compute z = x - y (mod 2^n) using the shares directly
   - Extract MSB: z_msb = (z >> (n-1)) & 1
   - Compute z' = z - (z_msb << (n-1))
   - Compute index: idx = 2^(n-1) - z' - 1
   - Evaluate: mb = EvalDDCF(key_b, idx)
   - Output: [[x ≥ y]]_b = b - (b · z_msb + mb - 2 · z_msb · mb) + r_b

**Correctness**: The arithmetic formula combines the MSB of z with the DDCF evaluation to produce shares that XOR to 1 if x ≥ y, and 0 otherwise.

**Performance Comparison**:
- **Key size**: DDCF keys are only ~⌈log n⌉ bits larger than DCF keys (negligible for n=32)
- **Communication**: DDCF approach requires NO communication during online phase (DCF approach needs one round)
- **Random values**: DDCF uses 3 (ρ₁, ρ₂, r), DCF uses 3 (α, r₀, r₁) - same amount
- **Computation**: DDCF approach evaluates once, DCF approach evaluates once - similar cost

The DDCF-based protocol is preferred in practice due to eliminating online communication.