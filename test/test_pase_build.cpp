#include "pase.h"
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <parser.h>


BOOST_AUTO_TEST_SUITE(TestPages)

    BOOST_AUTO_TEST_CASE(test_fill_dummy_pages)
    {
        size_t centroidTuplesPerPage = 204;
        BOOST_TEST(Page<CentroidTuple<float>>::calcTuplesSize() == centroidTuplesPerPage);

        PaseIVFFlat<float> pase(128, 20);
        Parser<float> parser("../../test/test_data/siftsmall_base.fvecs", 128, 10000);
        std::vector<std::vector<float>> parsed = parser.parse();
        std::vector<std::reference_wrapper<std::vector<float>>> postparsed(parsed.begin(), parsed.end());
        for (size_t i = 0; i < centroidTuplesPerPage + 10; ++i) {
            pase.addCentroid(postparsed, parsed[i]);
        }

        BOOST_TEST(pase.firstCentroidPage->tuples[0].vec == parsed[0]);

        auto *lastDataPage = pase.firstCentroidPage->tuples[0].firstDataPage;
        size_t dataPageCount = 0;
        while (lastDataPage->hasNextPage()) {
            lastDataPage = lastDataPage->nextPage;
            ++dataPageCount;
        }

        size_t vectorCountPerDataPage = Page<float>::calcTuplesSize() / parsed[0].size();
        size_t lastPageVectorCount = parsed.size() - vectorCountPerDataPage * dataPageCount;

        for (size_t i = 0; i < parsed[0].size(); ++i) {
            BOOST_TEST(lastDataPage->tuples[(lastPageVectorCount - 1) * parsed[0].size() + i] == parsed[parsed.size() - 1][i]);
        }

        auto nextCentroidPage = pase.firstCentroidPage->nextPage;

        for (size_t i = 0; i < 10; ++i) {
            BOOST_TEST(nextCentroidPage->tuples[i].vec == parsed[centroidTuplesPerPage + i]);
        }
    }

BOOST_AUTO_TEST_SUITE_END()