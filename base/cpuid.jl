# This file is a part of Julia. License is MIT: https://julialang.org/license

module CPUID

export cpu_isa

"""
    ISA(features::Set{UInt32})

A structure which represents the Instruction Set Architecture (ISA) of a
computer.  It holds the `Set` of features of the CPU.

Feature bit indices come from the cpufeatures library's generated tables
(extracted from LLVM's TableGen data at build time).
"""
struct ISA
    features::Set{UInt32}
end

Base.:<=(a::ISA, b::ISA) = a.features <= b.features
Base.:<(a::ISA,  b::ISA) = a.features <  b.features
Base.isless(a::ISA,  b::ISA) = a < b

include(string(Base.BUILDROOT, "features_h.jl"))  # include($BUILDROOT/base/features_h.jl)

"""
    _featurebytes_to_isa(buf::Vector{UInt8}) -> ISA

Convert a raw feature byte buffer (from cpufeatures) into an ISA.
"""
function _featurebytes_to_isa(buf::Vector{UInt8})
    features = Set{UInt32}()
    for byte_idx in 0:length(buf)-1
        b = buf[byte_idx + 1]
        b == 0 && continue
        for bit in 0:7
            if (b >> bit) & 1 != 0
                push!(features, UInt32(byte_idx * 8 + bit))
            end
        end
    end
    return ISA(features)
end

"""
    _cross_lookup_cpu(arch::String, name::String) -> ISA

Look up hardware features for a CPU on any architecture using the
cross-arch tables. Works regardless of host architecture.
Returns an empty ISA if the CPU or architecture is not found.
"""
function _cross_lookup_cpu(arch::String, name::String)
    nbytes = ccall(:jl_cpufeatures_cross_nbytes, Csize_t, (Cstring,), arch)
    nbytes == 0 && return ISA(Set{UInt32}())
    buf = Vector{UInt8}(undef, nbytes)
    written = ccall(:jl_cpufeatures_cross_lookup, Csize_t,
                    (Cstring, Cstring, Ptr{UInt8}, Csize_t),
                    arch, name, buf, nbytes)
    written == 0 && return ISA(Set{UInt32}())
    return _featurebytes_to_isa(buf)
end

"""
    _build_bit_to_name(arch::String) -> Dict{UInt32, String}

Build a mapping from feature bit index to feature name for an architecture.
"""
function _build_bit_to_name(arch::String)
    nfeats = ccall(:jl_cpufeatures_cross_num_features, UInt32, (Cstring,), arch)
    result = Dict{UInt32, String}()
    for i in 0:nfeats-1
        name_ptr = ccall(:jl_cpufeatures_cross_feature_name, Cstring, (Cstring, UInt32), arch, i)
        name_ptr == C_NULL && continue
        bit = ccall(:jl_cpufeatures_cross_feature_bit, Cint, (Cstring, UInt32), arch, i)
        bit < 0 && continue
        result[UInt32(bit)] = unsafe_string(name_ptr)
    end
    return result
end

"""
    feature_names(arch::String, cpu::String) -> Vector{String}
    feature_names(arch::String, isa::ISA) -> Vector{String}
    feature_names(isa::ISA) -> Vector{String}
    feature_names() -> Vector{String}

Return sorted hardware feature names. Can query by CPU name (on any
architecture) or by ISA. Defaults to the host architecture and CPU.

# Examples
```julia
feature_names()                           # host CPU features
feature_names("x86_64", "haswell")        # haswell's features
feature_names("aarch64", "cortex-x925")   # cross-arch query
```
"""
feature_names() = feature_names(string(Sys.ARCH), _host_isa())
feature_names(isa::ISA) = feature_names(string(Sys.ARCH), isa)
function feature_names(arch::String, cpu::String)
    isa = _cross_lookup_cpu(arch, cpu)
    return feature_names(arch, isa)
end
function feature_names(arch::String, isa::ISA)
    mapping = _build_bit_to_name(arch)
    return sort([get(mapping, bit, "unknown_$bit") for bit in isa.features])
end

"""
    _lookup_cpu(name::String) -> ISA

Look up hardware features for the named CPU on the host architecture.
Returns an empty ISA if the CPU name is not found.
"""
function _lookup_cpu(name::String)
    nbytes = ccall(:jl_cpufeatures_nbytes, Csize_t, ())
    buf = Vector{UInt8}(undef, nbytes)
    ret = ccall(:jl_cpufeatures_lookup, Cint, (Cstring, Ptr{UInt8}, Csize_t), name, buf, nbytes)
    ret != 0 && return ISA(Set{UInt32}())
    return _featurebytes_to_isa(buf)
end

"""
    _host_isa() -> ISA

Get the hardware features of the host CPU from the cpufeatures library.
"""
function _host_isa()
    nbytes = ccall(:jl_cpufeatures_nbytes, Csize_t, ())
    buf = Vector{UInt8}(undef, nbytes)
    ccall(:jl_cpufeatures_host, Cvoid, (Ptr{UInt8}, Csize_t), buf, nbytes)
    return _featurebytes_to_isa(buf)
end

# Build an ISA list for a given architecture family.
# Uses cross-arch lookup so it works on any host.
# Entries with empty cpuname get an empty ISA (generic baseline).
function _make_isa_list(arch::String, entries::Vector{Pair{String,String}})
    result = Pair{String,ISA}[]
    for (label, cpuname) in entries
        if isempty(cpuname)
            push!(result, label => ISA(Set{UInt32}()))
        else
            push!(result, label => _cross_lookup_cpu(arch, cpuname))
        end
    end
    return result
end

# ISA definitions per architecture family.
# CPU names are LLVM names in the cpufeatures database.
# Keep in sync with `arch_march_isa_mapping` in binaryplatforms.jl.
const ISAs_by_family = Dict(
    "i686" => _make_isa_list("x86_64", [
        "pentium4" => "",
        "prescott" => "prescott",
    ]),
    "x86_64" => _make_isa_list("x86_64", [
        "x86_64" => "",
        "core2" => "core2",
        "nehalem" => "nehalem",
        "sandybridge" => "sandybridge",
        "haswell" => "haswell",
        "skylake" => "skylake",
        "skylake_avx512" => "skylake-avx512",
    ]),
    "aarch64" => _make_isa_list("aarch64", [
        "armv8.0-a" => "",
        "armv8.1-a" => "cortex-a76",
        "armv8.2-a+crypto" => "cortex-a78",
        "a64fx" => "a64fx",
        "apple_m1" => "apple-a14",
    ]),
    "riscv64" => _make_isa_list("riscv64", [
        "riscv64" => "",
    ]),
)

# Test a CPU feature exists on the currently-running host
test_cpu_feature(feature::UInt32) = ccall(:jl_test_cpu_feature, Bool, (UInt32,), feature)

# Set of LLVM-canonical CPU feature names valid for the host architecture.
# Used only for parse-time validation in @cpu_supports — the macro discards
# the set after expansion so there is no runtime cost on the happy path.
# psABI levels (x86-64-vN) and named CPU models go through @cpu_uarch.
const _KNOWN_CPU_FEATURES = OncePerProcess{Set{Symbol}}() do
    s = Set{Symbol}()
    for (_, name) in _build_bit_to_name(string(Sys.ARCH))
        push!(s, Symbol(name))
    end
    s
end

# Look up the JIT-ready feature set for a CPU name on the host arch.
# Routes through jl_cpu_uarch_expand_features, which runs the same pipeline as
# multiversioning's `resolve_targets_for_llvm`: hw-masked and with
# non-deterministic features (rdrnd, rdseed, xsaveopt on x86) stripped, so
# the returned set matches what a sysimg clone targeting this CPU would
# actually have in its `target-features` attribute.
# Returns a sorted Vector{Symbol} of LLVM feature names, or `nothing` if the
# CPU name isn't known to LLVM for this architecture.
function _lookup_cpu_features(name::String)
    result = ccall(:jl_cpu_uarch_expand_features, Any, (Cstring,), name)
    result === nothing && return nothing
    feature_str = result::String
    isempty(feature_str) && return Symbol[]
    return sort!([Symbol(f) for f in split(feature_str, ',')])
end

@noinline _bad_feature_arg(x) =
    error("@cpu_supports: expected literal feature names (Symbol or String), got $(repr(x))")

# Parse a single macro argument as a feature Symbol.
function _parse_feature_arg(feature)
    if feature isa Symbol
        return feature
    elseif feature isa QuoteNode && feature.value isa Symbol
        return feature.value
    elseif feature isa AbstractString
        return Symbol(feature)
    else
        _bad_feature_arg(feature)
    end
end

"""
    Base.@cpu_supports feat1 [feat2 ...] -> Bool

Compile-time CPU feature query. Each `featN` must be a literal name
(Symbol or String) of an LLVM target feature. Multiple features combine
with logical AND: `@cpu_supports avx2 fma bmi2` is true iff *all* of them
are supported by the enclosing function's effective subtarget.

Mirrors GCC/Clang's `__builtin_cpu_supports`: each query is folded at
compile time using the enclosing function's effective `target-features`
and `target-cpu` attributes. Implications are honored — a function
compiled with `+avx512f` answers `@cpu_supports fma` as `true`.

Feature names must match LLVM's canonical spelling exactly. For names
containing `-` or `.` (which can't appear in a Julia identifier), pass a
String literal: `@cpu_supports "sse4.2"`, `@cpu_supports "amx-tile"`.

Feature names are validated at macro-expansion time; unknown names error
immediately.

For "do I have at least this CPU's feature set?" queries (including
x86-64 psABI levels like `x86-64-v3`), see [`Base.@cpu_uarch`](@ref).

# Examples
```julia
if @cpu_supports avx2
    # vectorize with 256-bit ops
end

if @cpu_supports avx2 fma bmi2 bmi
    # require this exact combination
end

if @cpu_supports "sse4.2"
    # hyphenated/dotted names need quoting
end
```
"""
macro cpu_supports(features...)
    isempty(features) && error("@cpu_supports: at least one feature name required")
    exprs = Expr[]
    for feature in features
        sym = _parse_feature_arg(feature)
        sym in _KNOWN_CPU_FEATURES() || error(
            "@cpu_supports: unknown CPU feature `$sym` for $(Sys.ARCH). ",
            "Use `Base.CPUID.feature_names()` to list features known to LLVM ",
            "for this architecture.")
        push!(exprs, :(Core.Intrinsics.cpu_supports($(QuoteNode(sym)))))
    end
    return foldr((a, b) -> :($a & $b), exprs)
end

"""
    Base.@cpu_uarch cpu -> Bool

Expand `cpu` (an LLVM CPU model name like `haswell`, `znver4`,
`x86-64-v3`, or `apple-m1`) into its JIT-ready feature set at
macro-expansion time, then check that the enclosing function's effective
subtarget supports all of them.

Names containing `-` need to be quoted as strings or Symbols:
`@cpu_uarch "apple-m1"` or `@cpu_uarch :apple_m1` (the latter via a
Symbol literal isn't possible in Julia syntax, but a String literal
works).

Useful for "do I have at least this CPU's capabilities?" queries —
including x86-64 psABI baselines (`x86-64-v2/v3/v4`). Implications and
feature supersets are picked up automatically because each individual
feature query is folded against the caller's subtarget.

The CPU name is looked up against the host architecture's LLVM CPU table
at macro-expansion time (via the same `resolve_targets_for_llvm` path
multiversioning uses); unknown names error immediately.

# Examples
```julia
if @cpu_uarch haswell
    # any CPU with at least Haswell's JIT-relevant features
end

if @cpu_uarch "x86-64-v3"
    # any CPU with at least the x86-64-v3 psABI baseline
end

if @cpu_uarch znver4
    # any CPU with at least Zen4's features
end
```
"""
macro cpu_uarch(cpu)
    sym = _parse_feature_arg(cpu)
    features = _lookup_cpu_features(String(sym))
    features === nothing && error(
        "@cpu_uarch: unknown CPU model `$sym` for $(Sys.ARCH).")
    isempty(features) && error(
        "@cpu_uarch: CPU model `$sym` has no hw features in the database; ",
        "this query would be vacuously true. Did you mean a more specific model?")
    exprs = [:(Core.Intrinsics.cpu_supports($(QuoteNode(f)))) for f in features]
    return foldr((a, b) -> :($a & $b), exprs)
end

# Normalize some variation in ARCH values (which typically come from `uname -m`)
function normalize_arch(arch::String)
    arch = lowercase(arch)
    if arch ∈ ("amd64",)
        arch = "x86_64"
    elseif arch ∈ ("i386", "i486", "i586")
        arch = "i686"
    elseif arch ∈ ("armv6",)
        arch = "armv6l"
    elseif arch ∈ ("arm", "armv7", "armv8", "armv8l")
        arch = "armv7l"
    elseif arch ∈ ("arm64",)
        arch = "aarch64"
    elseif arch ∈ ("ppc64le",)
        arch = "powerpc64le"
    end
    return arch
end

"""
    cpu_isa()

Return the [`ISA`](@ref) (instruction set architecture) of the current CPU.
"""
function cpu_isa()
    return _host_isa()
end

end # module CPUID
