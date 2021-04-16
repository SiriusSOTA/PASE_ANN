#include "parser.hpp"
#include "pase.hpp"
#include "utils.hpp"
#include "test_utils.hpp"
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>


BOOST_AUTO_TEST_SUITE(Profile)

    BOOST_AUTO_TEST_CASE(ANN_SIFT1M_PROFILE) {
        profileSift("../../test/test_data/sift/", "../../docs/our_profile/");
    }

    BOOST_AUTO_TEST_CASE(ANN_GIST1M_PROFILE) {
        profileGist("../../test/test_data/gist/", "../../docs/our_profile/");
    }

BOOST_AUTO_TEST_SUITE_END()