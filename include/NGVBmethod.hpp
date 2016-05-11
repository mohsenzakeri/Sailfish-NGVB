#ifndef __DUMM_HPP__
#define __DUMM_HPP__


#include <boost/math/special_functions/digamma.hpp>
#include <vector>
#include <cstdint>

class NGVBmethod {
public:
	NGVBmethod(int a);
private:
	int a_;
};
#endif