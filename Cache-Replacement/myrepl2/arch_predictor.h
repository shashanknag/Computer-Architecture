#ifndef HAWKEYE_PREDICTOR_H
#define HAWKEYE_PREDICTOR_H

using namespace std;
#include <vector>
#include <map>
#include "helper_function.h"

#define MAX_PCMAP 31
#define PCMAP_SIZE 2048
#define WEIGHT_PREV 0.99
#define WEIGHT_CURR 1 - WEIGHT_PREV
 


class Dist_Predictor{
private:
	map<uint64_t, uint64_t> PC_Map_dist, PC_Map_prev_time;

public:
	//Return prediction for PC Address
	int get_dist(uint64_t PC){
		uint64_t result = CRC(PC) % PCMAP_SIZE;
		return PC_Map_dist[result];
	}

	void update_dist(uint64_t PC, uint64_t curr_time){

		uint64_t result = CRC(PC) % PCMAP_SIZE;
		uint64_t curr_dist = curr_time - PC_Map_prev_time[result];
		PC_Map_dist[result] = PC_Map_dist[result]*WEIGHT_PREV + curr_dist*WEIGHT_CURR;
		PC_Map_prev_time[result] = curr_time;

	}
};


#endif
