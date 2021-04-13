#include <boost/test/unit_test.hpp>
#include <parser.hpp>
#include <cstdlib>

BOOST_AUTO_TEST_SUITE(TestParser)

    BOOST_AUTO_TEST_CASE(TestParseFloat)
    {
        Parser<float> parser("../../test/test_data/sift_small/siftsmall_base.fvecs", 128, 10000);
        std::vector<std::vector<float>> parsed = parser.parse();

        BOOST_TEST(parsed.size() == 10000);
        BOOST_TEST(parsed[0].size() == 128);
        BOOST_TEST(parsed[9999].size() == 128);

        BOOST_TEST(parsed[0][0] == 0);
        BOOST_TEST(parsed[0][1] == 16);
        BOOST_TEST(parsed[0][2] == 35);

        BOOST_TEST(parsed[9999][125] == 0);
        BOOST_TEST(parsed[9999][126] == 0);
        BOOST_TEST(parsed[9999][127] == 7);
    }


    BOOST_AUTO_TEST_CASE(TestParseChar)
    {
        Parser<char> parser("../../test/test_data/bigann_query.bvecs", 128, 10000);
        auto parsed = parser.parse();
        BOOST_TEST(parsed.size() == 10000);
        BOOST_TEST(parsed[0].size() == 128);
        BOOST_TEST(parsed[9999].size() == 128);

        BOOST_TEST(parsed[0][0] == 3);
        BOOST_TEST(parsed[0][1] == 9);
        BOOST_TEST(parsed[0][2] == 17);

        BOOST_TEST(parsed[9999][125] == 39);
        BOOST_TEST(parsed[9999][126] == 16);
        BOOST_TEST(parsed[9999][127] == 0);
    }

BOOST_AUTO_TEST_SUITE_END()
