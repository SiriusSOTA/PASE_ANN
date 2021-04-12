#pragma once

#ifdef __SSE3__
#include <immintrin.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif


inline float fvecL2sqrRef(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

inline float fvecNormL2sqrRef(const float* x, size_t d) {
    size_t i;
    double res = 0;
    for (i = 0; i < d; i++)
        res += x[i] * x[i];
    return res;
}

float fvec_inner_product_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++)
        res += x[i] * y[i];
    return res;
}

#ifdef __SSE3__

// reads 0 <= d < 4 floats as __m128
static inline __m128 maskedRead(int d, const float* x) {
    assert(0 <= d && d < 4);
//    ALIGNED(16)
    float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

inline float fvecNormL2sqr(const float* x, size_t d) {
    __m128 mx;
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        mx = _mm_loadu_ps(x);
        x += 4;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, mx));
        d -= 4;
    }

    mx = maskedRead(d, x);
    msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, mx));

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

#endif

#ifdef __AVX__

float fvecInnerProduct(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

float fvecL2sqr(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(a_m_b1, a_m_b1));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}


#elif defined(__SSE3__) // But not AVX

inline float fvecL2sqr(const float* x, const float* y, size_t d) {
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        // add the last 1, 2 or 3 values
        __m128 mx = maskedRead(d, x);
        __m128 my = maskedRead(d, y);
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

inline float fvecInnerProduct(const float* x, const float* y, size_t d) {
    __m128 mx, my;
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        mx = _mm_loadu_ps(x);
        x += 4;
        my = _mm_loadu_ps(y);
        y += 4;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, my));
        d -= 4;
    }

    // add the last 1, 2, or 3 values
    mx = maskedRead(d, x);
    my = maskedRead(d, y);
    __m128 prod = _mm_mul_ps(mx, my);

    msum1 = _mm_add_ps(msum1, prod);

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

#elif defined(__aarch64__)

float fvecL2sqr(const float* x, const float* y, size_t d) {
    if (d & 3)
        return fvecL2sqr_ref(x, y, d);
    float32x4_t accu = vdupq_n_f32(0);
    for (size_t i = 0; i < d; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t yi = vld1q_f32(y + i);
        float32x4_t sq = vsubq_f32(xi, yi);
        accu = vfmaq_f32(accu, sq, sq);
    }
    float32x4_t a2 = vpaddq_f32(accu, accu);
    return vdups_laneq_f32(a2, 0) + vdups_laneq_f32(a2, 1);
}

float fvecInnerProduct(const float* x, const float* y, size_t d) {
    if (d & 3)
        return fvecInnerProductRef(x, y, d);
    float32x4_t accu = vdupq_n_f32(0);
    for (size_t i = 0; i < d; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t yi = vld1q_f32(y + i);
        accu = vfmaq_f32(accu, xi, yi);
    }
    float32x4_t a2 = vpaddq_f32(accu, accu);
    return vdups_laneq_f32(a2, 0) + vdups_laneq_f32(a2, 1);
}

float fvecNormL2sqr(const float* x, size_t d) {
    if (d & 3)
        return fvecNormL2sqrRef(x, d);
    float32x4_t accu = vdupq_n_f32(0);
    for (size_t i = 0; i < d; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        accu = vfmaq_f32(accu, xi, xi);
    }
    float32x4_t a2 = vpaddq_f32(accu, accu);
    return vdups_laneq_f32(a2, 0) + vdups_laneq_f32(a2, 1);
}

#else
// scalar implementation

float fvecL2sqr(const float* x, const float* y, size_t d) {
    return fvecL2sqrRef(x, y, d);
}

float fvecInnerProduct(const float* x, const float* y, size_t d) {
    return fvecInnerProductRef(x, y, d);
}

float fvecNormL2sqr(const float* x, size_t d) {
    return fvecNormL2sqrRef(x, d);
}
#endif
