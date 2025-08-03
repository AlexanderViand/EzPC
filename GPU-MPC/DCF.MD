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