import ctypes.util
import os
import re
import sys

import cffi
import numpy as np

# 1. CRITICAL: Set dlopen flags to allow CUDA symbols to resolve globally
if (
    hasattr(sys, "setdlopenflags")
    and hasattr(os, "RTLD_NOW")
    and hasattr(os, "RTLD_GLOBAL")
):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

ffi = cffi.FFI()

# =============================================================================
# C Definition
# =============================================================================
ffi.cdef("""
    typedef struct {
        int resourceNumber;
        char* resourceName;
        char* implName;
        char* implDescription;
        long flags;
    } BeagleInstanceDetails;

    typedef struct {
       char* name;
       char* description;
       long supportFlags;
       long requiredFlags;
    } BeagleResource;

    typedef struct {
       BeagleResource* list;
       int length;
    } BeagleResourceList;

    typedef struct {
        int destinationPartials;
        int destinationScaleWrite;
        int destinationScaleRead;
        int child1Partials;
        int child1TransitionMatrix;
        int child2Partials;
        int child2TransitionMatrix;
    } BeagleOperation;

    BeagleResourceList* beagleGetResourceList(void);
    const char* beagleGetVersion(void);

    int beagleCreateInstance(
        int tipCount, int partialsBufferCount, int compactBufferCount,
        int stateCount, int patternCount, int eigenBufferCount,
        int matrixBufferCount, int categoryCount, int scaleBufferCount,
        int* resourceList, int resourceCount, long preferenceFlags,
        long requirementFlags, BeagleInstanceDetails* returnInfo);

    int beagleFinalizeInstance(int instance);

    int beagleSetTipStates(int instance, int tipIndex, const int* inStates);
    int beagleSetPartials(int instance, int bufferIndex, const double* inPartials);
    int beagleGetPartials(int instance, int bufferIndex, int scaleIndex, double* outPartials);

    int beagleSetStateFrequencies(int instance, int stateFrequenciesIndex, const double* inStateFrequencies);
    
    int beagleSetEigenDecomposition(int instance, int eigenIndex, const double* inEigenVectors, const double* inInverseEigenVectors, const double* inEigenValues);
    
    int beagleSetCategoryRates(int instance, const double* inCategoryRates);
    int beagleSetCategoryWeights(int instance, int categoryWeightsIndex, const double* inCategoryWeights);
    
    int beagleSetPatternWeights(int instance, const double* inPatternWeights);

    int beagleUpdateTransitionMatrices(int instance, int eigenIndex, const int* probabilityIndices, const int* firstDerivativeIndices, const int* secondDerivativeIndices, const double* edgeLengths, int count);
    
    int beagleSetTransitionMatrices(int instance, const int* matrixIndices, const double* inMatrices, const double* paddedValues, int count);
    
    int beagleGetTransitionMatrix(int instance, int matrixIndex, double* outMatrix);
    
    int beagleResetScaleFactors(int instance, int cumulativeScaleIndex);
    int beagleGetScaleFactors(int instance, int scaleIndex, double* outScaleFactors);

    int beagleUpdatePartials(const int instance, const BeagleOperation* operations, int operationCount, int cumulativeScaleIndex);
    int beagleUpdatePrePartials(const int instance, const BeagleOperation* operations, int operationCount, int cumulativeScaleIndex);
    
    int beagleCalculateRootLogLikelihoods(int instance, const int* bufferIndices, const int* categoryWeightsIndices, const int* stateFrequenciesIndices, const int* cumulativeScaleIndices, int count, double* outSumLogLikelihood);
    
    int beagleCalculateEdgeLogLikelihoods(
        int instance,
        const int* parentBufferIndices,
        const int* childBufferIndices,
        const int* probabilityIndices,
        const int* firstDerivativeIndices,
        const int* secondDerivativeIndices,
        const int* categoryWeightsIndices,
        const int* stateFrequenciesIndices,
        const int* cumulativeScaleIndices,
        int count,
        double* outSumLogLikelihood,
        double* outSumFirstDerivative,
        double* outSumSecondDerivative
    );
""")


# =============================================================================
# Library Loading
# =============================================================================
def _load_library():
    custom_path = os.environ.get("BEAGLE_LIB_PATH")
    if custom_path:
        return ffi.dlopen(custom_path)

    lib_names = ["libhmsbeagle.so", "libhmsbeagle.dylib", "hmsbeagle.dll"]
    home = os.environ.get("HOME")
    if home:
        local_path = os.path.join(home, "beagle", "lib", "libhmsbeagle.so")
        if os.path.exists(local_path):
            return ffi.dlopen(local_path)

    found_path = ctypes.util.find_library("hmsbeagle")
    if found_path:
        try:
            return ffi.dlopen(found_path)
        except OSError:
            pass

    for name in lib_names:
        try:
            return ffi.dlopen(name)
        except OSError:
            continue

    raise OSError(
        "Could not load Beagle library. Please set BEAGLE_LIB_PATH or LD_LIBRARY_PATH."
    )


lib = _load_library()
_MIN_BEAGLE_VERSION = "4.0"


def _parse_version_string(version_str):
    if not version_str:
        return ()
    tokens = re.findall(r"\d+", version_str)
    return tuple(int(tok) for tok in tokens)


def _ensure_minimum_version(lib, minimum):
    min_tuple = _parse_version_string(minimum)
    try:
        version_ptr = lib.beagleGetVersion()
    except AttributeError as exc:
        raise RuntimeError("Loaded BEAGLE library is missing beagleGetVersion") from exc
    if version_ptr == ffi.NULL:
        raise RuntimeError(
            "beagleGetVersion returned NULL; cannot verify BEAGLE version"
        )
    version_str = ffi.string(version_ptr).decode("ascii", errors="ignore")
    cur_tuple = _parse_version_string(version_str)
    if not cur_tuple:
        raise RuntimeError(f"Unable to parse BEAGLE version string '{version_str}'")
    max_len = max(len(cur_tuple), len(min_tuple))
    cur_tuple = cur_tuple + (0,) * (max_len - len(cur_tuple))
    min_tuple = min_tuple + (0,) * (max_len - len(min_tuple))
    if cur_tuple < min_tuple:
        raise RuntimeError(
            f"BEAGLE >= {minimum} required, but found version '{version_str}'. "
            "Please install a newer BEAGLE release."
        )


_ensure_minimum_version(lib, _MIN_BEAGLE_VERSION)


# =============================================================================
# Wrapper Class
# =============================================================================
class BeagleLikelihoodCalculator:
    def __init__(self, tip_count, state_count=4, pattern_count=1, use_gpu=False):
        self.tip_count = tip_count
        self.state_count = state_count
        self.pattern_count = pattern_count

        self._keep_alive = []
        self.pi = None

        self.node_count = 2 * tip_count - 1
        self._postorder_buffer_extra = 2
        self.postorder_capacity = self.node_count + self._postorder_buffer_extra
        self.preorder_offset = self.postorder_capacity
        self.partials_buffer_count = (
            self.preorder_offset + self.node_count + self._postorder_buffer_extra
        )
        self.matrix_buffer_count = self.node_count + 2
        self.scale_count = self.node_count + 2
        self.cumulative_scale_index = 0

        flags = 2 | 128
        requirement_flags = 0

        if use_gpu:
            print("[DEBUG] Requesting GPU (Auto-detect)", flush=True)
            resource_ptr = ffi.NULL
            resource_count = 0
        else:
            self._cpu_res_list = ffi.new("int[]", [0])
            resource_ptr = self._cpu_res_list
            resource_count = 1

        details = ffi.new("BeagleInstanceDetails*")

        self.inst = lib.beagleCreateInstance(
            tip_count,
            self.partials_buffer_count,
            0,
            state_count,
            pattern_count,
            1,
            self.matrix_buffer_count,
            1,
            self.scale_count,
            resource_ptr,
            resource_count,
            flags,
            requirement_flags,
            details,
        )

        if self.inst < 0:
            self._diagnose_resources()
            raise RuntimeError(f"Beagle CreateInstance failed with code {self.inst}")

        try:
            self.resource_name = ffi.string(details.resourceName).decode("utf-8")
        except:
            self.resource_name = "Unknown"

        self._init_pattern_weights()

    def set_pattern_weights(self, weights=None):
        if weights is None:
            self.pattern_weights = np.ones(self.pattern_count, dtype=np.float64)
        else:
            if len(weights) != self.pattern_count:
                raise ValueError(
                    f"Weights length {len(weights)} != pattern count {self.pattern_count}"
                )
            self.pattern_weights = np.array(weights, dtype=np.float64)

        c_weights = self._to_aligned_c_array(self.pattern_weights, "double*")
        code = lib.beagleSetPatternWeights(self.inst, c_weights)
        self._check(code, "beagleSetPatternWeights")

    def _init_pattern_weights(self):
        self.set_pattern_weights(None)

    def _diagnose_resources(self):
        rlist = lib.beagleGetResourceList()
        if not rlist:
            print("  No resource list returned.")
            return
        for i in range(rlist.length):
            res = rlist.list[i]
            name = ffi.string(res.name).decode("utf-8")
            print(f"  Resource {i}: {name}")

    def _check(self, code, name):
        if code < 0:
            raise RuntimeError(f"Beagle call '{name}' failed with error code {code}")

    def _to_aligned_c_array(self, data, c_type="double*", alignment=32):
        if isinstance(data, (list, tuple)):
            dtype = np.int32 if "int" in c_type else np.float64
            data = np.array(data, dtype=dtype)

        if isinstance(data, np.ndarray):
            if "int" in c_type and data.dtype != np.int32:
                data = data.astype(np.int32)
            elif "double" in c_type:
                if np.iscomplexobj(data):
                    data = np.real(data).astype(np.float64)
                elif data.dtype != np.float64:
                    data = data.astype(np.float64)

            flat_data = np.ascontiguousarray(data.flatten())
            byte_data = flat_data.tobytes()
            nbytes = flat_data.nbytes
        else:
            byte_data = data
            nbytes = len(data)

        raw_storage = ffi.new("char[]", nbytes + alignment)
        raw_addr = int(ffi.cast("uintptr_t", raw_storage))
        offset = alignment - (raw_addr % alignment)
        if offset == alignment:
            offset = 0

        aligned_ptr = raw_storage + offset
        target_buf = ffi.buffer(aligned_ptr, nbytes)
        target_buf[:] = byte_data
        final_ptr = ffi.cast(c_type, aligned_ptr)

        self._keep_alive.append(raw_storage)
        return final_ptr

    def set_model_matrix(self, Q, p_root):
        self.pi = np.array(p_root, dtype=np.float64)
        evals, V = np.linalg.eig(Q)
        V_inv = np.linalg.inv(V)

        c_V = self._to_aligned_c_array(V, "double*")
        c_Vinv = self._to_aligned_c_array(V_inv, "double*")
        c_evals = self._to_aligned_c_array(evals, "double*")

        code = lib.beagleSetEigenDecomposition(self.inst, 0, c_V, c_Vinv, c_evals)
        self._check(code, "beagleSetEigenDecomposition")

        c_freqs = self._to_aligned_c_array(p_root, "double*")
        code = lib.beagleSetStateFrequencies(self.inst, 0, c_freqs)
        self._check(code, "beagleSetStateFrequencies")

        c_rates = self._to_aligned_c_array(np.array([1.0, 1.0, 1.0, 1.0]), "double*")
        code = lib.beagleSetCategoryRates(self.inst, c_rates)
        self._check(code, "beagleSetCategoryRates")

        c_weights = self._to_aligned_c_array(np.array([1.0, 0.0, 0.0, 0.0]), "double*")
        code = lib.beagleSetCategoryWeights(self.inst, 0, c_weights)
        self._check(code, "beagleSetCategoryWeights")

    def set_tip_partials(self, tip_dict):
        for idx, partials in tip_dict.items():
            c_partials = self._to_aligned_c_array(partials, "double*")
            code = lib.beagleSetPartials(self.inst, idx, c_partials)
            self._check(code, f"beagleSetPartials(tip={idx})")

    def set_partials(self, buffer_index, partials):
        c_partials = self._to_aligned_c_array(partials, "double*")
        code = lib.beagleSetPartials(self.inst, buffer_index, c_partials)
        self._check(code, f"beagleSetPartials(buffer={buffer_index})")

    def get_partials(self, buffer_index, scale_index=-1):
        buffer_size = self.state_count * self.pattern_count
        out_array = np.zeros(buffer_size, dtype=np.float64)
        c_out = self._to_aligned_c_array(out_array, "double*")
        code = lib.beagleGetPartials(self.inst, buffer_index, scale_index, c_out)
        self._check(code, f"beagleGetPartials(buf={buffer_index})")
        raw_bytes = ffi.buffer(c_out, buffer_size * 8)
        return np.frombuffer(raw_bytes, dtype=np.float64).copy()

    def get_scale_factors(self, scale_index):
        size = self.pattern_count
        out_array = np.zeros(size, dtype=np.float64)
        c_out = self._to_aligned_c_array(out_array, "double*")
        code = lib.beagleGetScaleFactors(self.inst, scale_index, c_out)
        self._check(code, "beagleGetScaleFactors")
        raw_bytes = ffi.buffer(c_out, size * 8)
        return np.frombuffer(raw_bytes, dtype=np.float64).copy()

    def get_partials_and_scale(self, buffer_index):
        raw_partials = self.get_partials(buffer_index, scale_index=-1)
        partials = raw_partials.reshape(self.pattern_count, self.state_count)
        log_scales = self.get_scale_factors(buffer_index)
        return partials, log_scales

    def get_all_partials_and_scales(self, node_count):
        """
        Efficiently retrieves partials and scales for ALL nodes.
        Zeroes out scales for TIPS to avoid reading uninitialized memory.
        """
        all_partials = np.zeros(
            (node_count, self.pattern_count, self.state_count), dtype=np.float64
        )
        all_scales = np.zeros((node_count, self.pattern_count), dtype=np.float64)

        for i in range(node_count):
            # Get Partials
            p = self.get_partials(i, scale_index=-1)
            all_partials[i] = p.reshape(self.pattern_count, self.state_count)

            # Get Scales
            # Only INTERNAL nodes (indices >= tip_count) have valid scale factors
            # Tips have scale 0.0 (log scale 0.0 = unscaled)
            if i >= self.tip_count:
                s = self.get_scale_factors(i)
                all_scales[i] = s
            else:
                # Explicitly zero for tips (redundant due to np.zeros init, but explicit is safe)
                all_scales[i] = 0.0

        return all_partials, all_scales

    def get_transition_matrix(self, matrix_index):
        size = self.state_count * self.state_count
        out_array = np.zeros(size, dtype=np.float64)
        c_out = self._to_aligned_c_array(out_array, "double*")
        code = lib.beagleGetTransitionMatrix(self.inst, matrix_index, c_out)
        self._check(code, f"beagleGetTransitionMatrix(idx={matrix_index})")
        raw_bytes = ffi.buffer(c_out, size * 8)
        return (
            np.frombuffer(raw_bytes, dtype=np.float64)
            .copy()
            .reshape(self.state_count, self.state_count)
        )

    def update_transition_matrices(
        self,
        edge_lengths,
        probability_indices=None,
        first_derivative_indices=None,
        second_derivative_indices=None,
    ):
        count = len(edge_lengths)
        c_edge_lengths = self._to_aligned_c_array(edge_lengths, "double*")

        if probability_indices is None:
            probability_indices = np.arange(count, dtype=np.int32)

        c_prob_indices = self._to_aligned_c_array(probability_indices, "int*")

        if first_derivative_indices is not None:
            c_d1_indices = self._to_aligned_c_array(first_derivative_indices, "int*")
        else:
            c_d1_indices = ffi.NULL

        if second_derivative_indices is not None:
            c_d2_indices = self._to_aligned_c_array(second_derivative_indices, "int*")
        else:
            c_d2_indices = ffi.NULL

        code = lib.beagleUpdateTransitionMatrices(
            self.inst,
            0,
            c_prob_indices,
            c_d1_indices,
            c_d2_indices,
            c_edge_lengths,
            count,
        )
        self._check(code, "beagleUpdateTransitionMatrices")

    def set_transition_matrices(self, matrices, edge_count):
        c_matrices = self._to_aligned_c_array(matrices.flatten(), "double*")
        indices = np.arange(edge_count, dtype=np.int32)
        c_indices = self._to_aligned_c_array(indices, "int*")

        pad_size = edge_count * self.state_count * self.state_count
        padded_zeros = np.zeros(pad_size, dtype=np.float64)
        c_padded = self._to_aligned_c_array(padded_zeros, "double*")

        code = lib.beagleSetTransitionMatrices(
            self.inst, c_indices, c_matrices, c_padded, edge_count
        )
        self._check(code, "beagleSetTransitionMatrices")

    def prepare_operations(self, operations):
        op_dtype = np.dtype(
            [
                ("dest", "i4"),
                ("destScaleWrite", "i4"),
                ("destScaleRead", "i4"),
                ("child1", "i4"),
                ("child1Matrix", "i4"),
                ("child2", "i4"),
                ("child2Matrix", "i4"),
            ]
        )
        ops_array = np.zeros(len(operations), dtype=op_dtype)
        for i, op in enumerate(operations):
            ops_array[i]["dest"] = op["dest"]
            ops_array[i]["destScaleWrite"] = op.get("destScaleWrite", op["dest"])
            ops_array[i]["destScaleRead"] = op.get("destScaleRead", -1)
            ops_array[i]["child1"] = op["child1"]
            ops_array[i]["child1Matrix"] = op.get("child1Matrix", op["child1"])
            ops_array[i]["child2"] = op["child2"]
            ops_array[i]["child2Matrix"] = op.get("child2Matrix", op["child2"])

        return self._to_aligned_c_array(ops_array, "BeagleOperation*")

    def update_partials(self, operations, operation_count=None):
        code = lib.beagleResetScaleFactors(self.inst, self.cumulative_scale_index)
        self._check(code, "beagleResetScaleFactors")

        c_ops = operations
        count = operation_count

        if isinstance(operations, (list, tuple)):
            c_ops = self.prepare_operations(operations)
            count = len(operations)

        if count is None:
            raise ValueError(
                "operation_count required when passing pre-computed operations"
            )

        code = lib.beagleUpdatePartials(
            self.inst, c_ops, count, self.cumulative_scale_index
        )
        self._check(code, "beagleUpdatePartials")

    def update_pre_partials(self, operations, operation_count=None):
        code = lib.beagleResetScaleFactors(self.inst, self.cumulative_scale_index)
        self._check(code, "beagleResetScaleFactors")

        c_ops = operations
        count = operation_count

        if isinstance(operations, (list, tuple)):
            c_ops = self.prepare_operations(operations)
            count = len(operations)
        if count is None:
            raise ValueError(
                "operation_count required when passing pre-computed operations"
            )

        code = lib.beagleUpdatePrePartials(
            self.inst, c_ops, count, self.cumulative_scale_index
        )
        self._check(code, "beagleUpdatePrePartials")

    def calculate_site_log_likelihoods(self, root_index, pi=None):
        if pi is None:
            if self.pi is None:
                raise RuntimeError("Model matrix not set and pi not provided")
            pi = self.pi

        root_partials = self.get_partials(root_index)
        log_scalers = self.get_scale_factors(self.cumulative_scale_index)

        partials_matrix = root_partials.reshape(self.pattern_count, self.state_count)
        site_probs = np.dot(partials_matrix, pi)

        with np.errstate(divide="ignore"):
            log_site_probs = np.log(site_probs)

        return log_site_probs + log_scalers

    def calculate_root_log_likelihood(self, root_index):
        site_ll = self.calculate_site_log_likelihoods(root_index)
        return np.dot(site_ll, self.pattern_weights)

    def calculate_edge_derivatives(self, operations, edge_count):
        parent_indices = np.zeros(edge_count, dtype=np.int32)
        child_indices = np.zeros(edge_count, dtype=np.int32)
        prob_indices = np.arange(edge_count, dtype=np.int32)

        node_to_parent = {}
        for op in operations:
            node_to_parent[op["child1"]] = op["dest"]
            node_to_parent[op["child2"]] = op["dest"]

        for i in range(edge_count):
            child_indices[i] = i
            if i in node_to_parent:
                parent_indices[i] = node_to_parent[i]
            else:
                parent_indices[i] = i

        c_parent_indices = self._to_aligned_c_array(parent_indices, "int*")
        c_child_indices = self._to_aligned_c_array(child_indices, "int*")
        c_prob_indices = self._to_aligned_c_array(prob_indices, "int*")

        c_cat_weights = self._to_aligned_c_array([0], "int*")
        c_freqs = self._to_aligned_c_array([0], "int*")

        c_scales = self._to_aligned_c_array([self.cumulative_scale_index], "int*")

        out_logl = np.zeros(edge_count, dtype=np.float64)
        out_d1 = np.zeros(edge_count, dtype=np.float64)
        out_d2 = np.zeros(edge_count, dtype=np.float64)

        c_out_logl = self._to_aligned_c_array(out_logl, "double*")
        c_out_d1 = self._to_aligned_c_array(out_d1, "double*")
        c_out_d2 = self._to_aligned_c_array(out_d2, "double*")

        code = lib.beagleCalculateEdgeLogLikelihoods(
            self.inst,
            c_parent_indices,
            c_child_indices,
            c_prob_indices,
            ffi.NULL,
            ffi.NULL,
            c_cat_weights,
            c_freqs,
            c_scales,
            edge_count,
            c_out_logl,
            c_out_d1,
            c_out_d2,
        )
        self._check(code, "beagleCalculateEdgeLogLikelihoods")

        return np.frombuffer(
            ffi.buffer(c_out_d1, edge_count * 8), dtype=np.float64
        ).copy()

    def get_all_partials(self, node_count):
        all_p = []
        for i in range(node_count):
            all_p.append(self.get_partials(i))
        return np.array(all_p).reshape(node_count, self.pattern_count, self.state_count)

    def get_all_pre_partials(self, node_count):
        all_p = np.zeros(
            (node_count, self.pattern_count, self.state_count), dtype=np.float64
        )
        for i in range(node_count):
            buf_idx = self.preorder_buffer_index(i)
            raw = self.get_partials(buf_idx, scale_index=-1)
            all_p[i] = raw.reshape(self.pattern_count, self.state_count)
        return all_p

    def preorder_buffer_index(self, node_index):
        return self.preorder_offset + node_index

    def __del__(self):
        if hasattr(self, "inst") and self.inst >= 0:
            lib.beagleFinalizeInstance(self.inst)
