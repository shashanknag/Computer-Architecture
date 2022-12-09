/**
 * Title: CS6600, Computer Architecture, Assignment - 3
 * Author: Sai Gautham Ravipati (EE19B053), Shashank Nag (EE19B118) 
 * Description: Branch Prediction using 2-layered network
 **/

#include <algorithm>
#include <array>
#include <bitset>
#include <deque>
#include <map>

#include "ooo_cpu.h"

template <typename T, std::size_t HISTLEN, std::size_t BITS>

class pplus
{
  T bias1 = 0;       // Bias of first-stage output-1
  T bias2 = 0;       // Bias of first-stage output-2
  T bias3 = 0;       // Bias of first-stage output-3
  T bias4 = 0;       // Bias of first-stage output-4
  T bias = 0;        // Bias of second-stage output

  T output1 = 0;     // output of node-1
  T output2 = 0;     // output of node-2 
  T output3 = 0;     // output of node-3
  T output4 = 0;     // output of node-4
  T output = 0;      // output of second-stage 
  
  std::array<T, HISTLEN/8> weights1 = {};        // weights to compute node-1
  std::array<T, HISTLEN/4> weights2 = {};        // weights to compute node-2
  std::array<T, HISTLEN/2> weights3 = {};        // weights to compute node-3
  std::array<T, HISTLEN> weights4 = {};          // weights to compute node-4
  std::array<T, 4> weights = {};                 // weights to compute final-stage output 

public:
  // Maximum and Minimum weight values
  // Need to saturate weights according to the BIT_WIDTH
  constexpr static T max_weight = (1 << (BITS - 1)) - 1;
  constexpr static T min_weight = -(max_weight + 1);

  T predict(std::bitset<HISTLEN> history)
  {
    // Initialising output of each stage to the corresponding bias
    output1 = bias1;       
    output2 = bias2; 
    output3 = bias3; 
    output4 = bias4; 
    output = bias; 

    /* Each node of the perceptron considers a geometric length of 
     * the input history. Node-1 -> 4, Node -> 8, ... 
     */ 

    // Accumulating the activations of first-stage based on the inputs
    for (std::size_t i = 0; i < std::size(history)/8; i++) {
      if (history[i])
        output1 += weights1[i];
      else
        output1 -= weights1[i];
    }

    for (std::size_t i = 0; i < std::size(history)/4; i++) {
      if (history[i])
        output2 += weights2[i];
      else
        output2 -= weights2[i];
    }

    for (std::size_t i = 0; i < std::size(history)/2; i++) {
      if (history[i])
        output3 += weights3[i];
      else
        output3 -= weights3[i];
    }

    for (std::size_t i = 0; i < std::size(history); i++) {
      if (history[i])
        output4 += weights4[i];
      else
        output4 -= weights4[i];
    }

    // Computing the output activation using the activations of first layers
    output += weights[0] * output1 + weights[1] * output2 + weights[2] * output3 + weights[3] * output4;

    return output;
  }

  void update(bool result, std::bitset<HISTLEN> history)
  {
    // Updating the bias-value based on the branch result
    if (result)
      bias = std::min(bias + 1, max_weight);
    else
      bias = std::max(bias - 1, min_weight);

    // Updating the weights of second stage, using the intermediate result
    if(result == (output1 >= 0))
      weights[0] = std::min(weights[0] + 1, max_weight);
    else 
      weights[0] = std::max(weights[0] - 1, min_weight);

    if(result == (output2 >= 0))
      weights[1] = std::min(weights[1] + 1, max_weight);
    else 
      weights[1] = std::max(weights[1] - 1, min_weight);

    if(result == (output3 >= 0))
      weights[2] = std::min(weights[2] + 1, max_weight);
    else 
      weights[2] = std::max(weights[2] - 1, min_weight);

    if(result == (output4 >= 0))
      weights[3] = std::min(weights[3] + 1, max_weight);
    else 
      weights[3] = std::max(weights[3] - 1, min_weight);

    /* For each weight and corresponding bit in the history register. 
     * Updates the i'th bit in the history positively if it correlates 
     * with this branch outcome.
     */
    auto upd_mask = result ? history : ~history; 

    // Updating the weights of first node, using the global history mask  
    // Updating the bias-value based on the branch result
    if(result != (output1 >= 0)){
      if (result)
        bias1 = std::min(bias1 + 1, max_weight);
      else
        bias1 = std::max(bias1 - 1, min_weight);
                                                                                                                             
      for (std::size_t i = 0; i < std::size(upd_mask)/8; i++) {
        if (upd_mask[i])
          weights1[i] = std::min(weights1[i] + 1, max_weight);
        else
          weights1[i] = std::max(weights1[i] - 1, min_weight);
      }
    }
    // Updating the weights of second node, using the global history mask  
    // Updating the bias-value based on the branch result
    if(result != (output2 >= 0)){
      if (result)
        bias2 = std::min(bias2 + 1, max_weight);
      else
        bias2 = std::max(bias2 - 1, min_weight);
                                                                                                                             
      for (std::size_t i = 0; i < std::size(upd_mask)/4; i++) {
        if (upd_mask[i])
          weights2[i] = std::min(weights2[i] + 1, max_weight);
        else
          weights2[i] = std::max(weights2[i] - 1, min_weight);
      }
    }

    // Updating the weights of third node, using the global history mask  
    // Updating the bias-value based on the branch result
    if(result != (output3 >= 0)){
      if (result)
        bias3 = std::min(bias3 + 1, max_weight);
      else
        bias3 = std::max(bias3 - 1, min_weight);
                                                                                                                             
      for (std::size_t i = 0; i < std::size(upd_mask)/2; i++) {
        if (upd_mask[i])
          weights3[i] = std::min(weights3[i] + 1, max_weight);
        else
          weights3[i] = std::max(weights3[i] - 1, min_weight);
      }
    }

    // Updating the weights of fourth node, using the global history mask  
    // Updating the bias-value based on the branch result
    if(result != (output4 >= 0)){
      if (result)
        bias4 = std::min(bias4 + 1, max_weight);
      else
        bias4 = std::max(bias4 - 1, min_weight);
                                                                                                                             
      for (std::size_t i = 0; i < std::size(upd_mask); i++) {
        if (upd_mask[i])
          weights4[i] = std::min(weights4[i] + 1, max_weight);
        else
          weights4[i] = std::max(weights4[i] - 1, min_weight);
      }
    }
  }
};

constexpr std::size_t PPLUS_HISTORY = 32; // History length for the global history shift register
constexpr std::size_t PPLUS_BITS = 8;     // Number of bits per weight
constexpr std::size_t NUM_PPLUSS = 163;   // No. of entries in the history table 

constexpr int THETA = 1.93 * PPLUS_HISTORY + 14; // Threshold for training

constexpr std::size_t NUM_UPDATE_ENTRIES = 100;  // Size of buffer for keeping 'pplus_state' for update

/* 'pplus_state' - stores the branch prediction and keeps information
 * such as output and history needed for updating the pplus predictor
 */
struct pplus_state {
  uint64_t ip = 0;
  bool prediction = false;                     // prediction: 1 for taken, 0 for not taken
  int output = 0;                              // pplus output
  std::bitset<PPLUS_HISTORY> history = 0; // value of the history register yielding this prediction
};

std::map<O3_CPU*, std::array<pplus<int, PPLUS_HISTORY, PPLUS_BITS>,
                             NUM_PPLUSS>> ppluss;             // table of ppluss
std::map<O3_CPU*, std::deque<pplus_state>> pplus_state_buf;   // state for updating pplus predictor
std::map<O3_CPU*, std::bitset<PPLUS_HISTORY>> spec_global_history; // speculative global history - updated by predictor
std::map<O3_CPU*, std::bitset<PPLUS_HISTORY>> global_history;      // real global history - updated when the predictor is
                                                                        // updated

void O3_CPU::initialize_branch_predictor() {}

int direction[1]; 

uint8_t O3_CPU::predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type)
{
  // Hash the address to get an index into the table of srnns
  direction[0] = predicted_target - ip;
  auto index = ip % NUM_PPLUSS;
  index += direction[0] % NUM_PPLUSS;
  auto output = ppluss[this][index].predict(spec_global_history[this]);

  bool prediction = (output >= 0);

  // Record the various values needed to update the predictor
  pplus_state_buf[this].push_back({ip, prediction, output, spec_global_history[this]});
  if (std::size(pplus_state_buf[this]) > NUM_UPDATE_ENTRIES)
    pplus_state_buf[this].pop_front();

  // Update the speculative global history register
  spec_global_history[this] <<= 1;
  spec_global_history[this].set(0, prediction);
  return prediction;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
  auto state = std::find_if(std::begin(pplus_state_buf[this]), std::end(pplus_state_buf[this]), [ip](auto x) { return x.ip == ip; });
  if (state == std::end(pplus_state_buf[this]))
    return; // Skip update because state was lost

  auto [_ip, prediction, output, history] = *state;
  pplus_state_buf[this].erase(state);

  auto index = ip % NUM_PPLUSS;
  index += direction[0] % NUM_PPLUSS; 

  // update the real global history shift register
  global_history[this] <<= 1;
  global_history[this].set(0, taken);

  /* If this branch was mispredicted, restore the speculative history to the
   * last known real history.
   */
  if (prediction != taken)
    spec_global_history[this] = global_history[this];

  /* if the output of the srnn predictor is outside of the range
   * [-theta,theta] *and* the prediction was correct, then we don't need to
   * adjust the weights.
   */
  if ((output <= THETA && output >= -THETA) || (prediction != taken))
    ppluss[this][index].update(taken, history);
}