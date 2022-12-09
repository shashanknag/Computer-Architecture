/**
 * Title: CS6600, Computer Architecture, Assignment - 3
 * Author: Sai Gautham Ravipati (EE19B053), Shashank Nag (EE19B118) 
 * Description: Branch Prediction using a CNN with 1-filter
 **/

// Required for updating the value of theta
#define SPEED 18

#include <algorithm>
#include <array>
#include <bitset>
#include <deque>
#include <map>

#include "ooo_cpu.h"

template <typename T, std::size_t HISTLEN, std::size_t BITS>

class cnn
{
  T output = 0;                         // Stores the output of CNN
  T bias = 0;                           // Stores the bias to be added to the output 
  
  std::array<T, 4> W0 = {};                  // Filter-1 weights 
  std::array<T, HISTLEN - 4> A = {};         // Holds the convolved output of Filter-1
  std::array<T, HISTLEN - 4> L1 = {};        // Holds the weights of second layer node-1

public:
  // maximum and minimum weight values
  constexpr static T max_weight = (1 << (BITS - 1)) - 1;
  constexpr static T min_weight = -(max_weight + 1);

  T predict(std::bitset<HISTLEN> history)
  {
    
    output = 0;       // Stores the output of CNN

    // Initialising the activations of second layer to 0
    for (std::size_t i = 0; i < std::size(history) - 4; i++){
      A[i] = 0;              
    }

    // Performing 1D convolution on the global history to obtain the output
    // Stride of 1, and Filter-width being considered is 4. 
    for (std::size_t i = 0; i < std::size(history) - 4; i++) {
      if (history[i]){
        A[i] += W0[0];
      }
      else{
        A[i] -= W0[0];
      }

      if(history[i + 1]){
        A[i] += W0[1];
      }
      else{
        A[i] -= W0[1];
      }

      if (history[i + 2]){
        A[i] += W0[2];
      }
      else{
        A[i] -= W0[2];
      }

      if (history[i + 3]){
        A[i] += W0[3];
      }
      else{
        A[i] -= W0[4];
      }
    }

    // Using the activations of first layer to compute the output
    for (std::size_t i = 0; i < std::size(history) - 4; i++) {
          output += L1[i] * A[i]; 
    }
    output += bias; 

    return output;
  }

  void update(bool result, std::bitset<HISTLEN> history)
  {

    // Updating the bias-value based on the branch result
    if(result)  
      bias = std::min(bias + 1, max_weight);
    else
      bias = std::max(bias - 1, min_weight);

    /* For each weight and corresponding bit in the history register. 
     * Updates the i'th bit in the history positively if it correlates 
     * with this branch outcome.
     */
    auto upd_mask = result ? history : ~history; 

    /* Updating the weights of second level using intermediate activations 
     * If the intermediate activation correlates with the branch, the weight 
     * is increased.
     */ 
    for (std::size_t i = 0; i < std::size(upd_mask) - 4; i++) {
      if(result == (A[i] >= 0))
        L1[i] = std::min(L1[i] + 1, max_weight); 
      else
        L1[i] = std::max(L1[i] - 1, min_weight); 
    } 

    // Updating the filter-weights based on the input global history
    for (std::size_t i = 0; i < std::size(upd_mask) - 4; i++) {
      if(upd_mask[i])
        W0[0] = std::min(W0[0] + 1, max_weight); 
      else
        W0[0] = std::max(W0[0] - 1, min_weight); 

      if(upd_mask[i + 1])
        W0[1] = std::min(W0[1] + 1, max_weight); 
      else
        W0[1] = std::max(W0[1] - 1, min_weight); 

      if(upd_mask[i + 2])
        W0[2] = std::min(W0[2] + 1, max_weight); 
      else
        W0[2] = std::max(W0[2] - 1, min_weight); 

      if(upd_mask[i + 3])
        W0[3] = std::min(W0[3] + 1, max_weight); 
      else
        W0[3] = std::max(W0[3] - 1, min_weight); 
      }

  }
};

constexpr std::size_t CNN_HISTORY = 32;                  // History length for the global history shift register
constexpr std::size_t CNN_BITS = 8;                      // Number of bits per weight
constexpr std::size_t NUM_CNNS = 2048;                   // Number of entries required in the history table 
constexpr int THETA = 1.73 * CNN_HISTORY + 14;           // Threshold for training
constexpr std::size_t NUM_UPDATE_ENTRIES = 100;          // Size of buffer for keeping 'cnn_state' for update

/* 'cnn_state' - stores the branch prediction and keeps information
 * such as output and history needed for updating the cnn predictor
 */
struct cnn_state {
  uint64_t ip = 0;
  bool prediction = false;                       // prediction: 1 for taken, 0 for not taken
  int output = 0;                                // cnn output
  std::bitset<CNN_HISTORY> history = 0;   // value of the history register yielding this prediction
};

std::map<O3_CPU*, std::array<cnn<int, CNN_HISTORY, CNN_BITS>,
                             NUM_CNNS>> cnns;             // table of cnns
std::map<O3_CPU*, std::deque<cnn_state>> cnn_state_buf;   // state for updating cnn predictor
std::map<O3_CPU*, std::bitset<CNN_HISTORY>> spec_global_history; // speculative global history - updated by predictor
std::map<O3_CPU*, std::bitset<CNN_HISTORY>> global_history;      // real global history - updated when the predictor is
                                                                        // updated

int direction[1];       // Used in hashing to consider the direction of branch based on ip and target 
// Placeholder for dynamic threshold setting 
int theta[1]; 
int tc[1]; 

void O3_CPU::initialize_branch_predictor() {
  tc[0] = 0;
  theta[0] = THETA; 
}

uint8_t O3_CPU::predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type)
{
  // Hash the address to get an index into the table of cnns
  direction[0] = predicted_target - ip;
  auto index = ip ^ (ip >> CNN_HISTORY) ^ (int)(spec_global_history[this].to_ulong());
  index = index % NUM_CNNS;
  index += direction[0] % NUM_CNNS;
  auto output = cnns[this][index].predict(spec_global_history[this]);

  bool prediction = (output >= 0);

  // Record the various values needed to update the predictor
  cnn_state_buf[this].push_back({ip, prediction, output, spec_global_history[this]});
  if (std::size(cnn_state_buf[this]) > NUM_UPDATE_ENTRIES)
    cnn_state_buf[this].pop_front();

  // Update the speculative global history register
  spec_global_history[this] <<= 1;
  spec_global_history[this].set(0, prediction);
  return prediction;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
  auto state = std::find_if(std::begin(cnn_state_buf[this]), std::end(cnn_state_buf[this]), [ip](auto x) { return x.ip == ip; });
  if (state == std::end(cnn_state_buf[this]))
    return; // Skip update because state was lost

  auto [_ip, prediction, output, history] = *state;
  cnn_state_buf[this].erase(state);

  auto index = ip ^ (ip >> CNN_HISTORY) ^ (int)(history.to_ulong());
  index = index % NUM_CNNS;
  index += direction[0] % NUM_CNNS; 

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
  if ((output <= theta[0] && output >= -theta[0]) || (prediction != taken)){
    cnns[this][index].update(taken, history);
    int a = output <= 0 ? -output : output; 

    // Dynamic threshold setting from Seznec's O-GEHL paper
    if (prediction != taken) {
      // Increase theta after enough mispredictions
      tc[0]++;
      if (tc[0] >= SPEED) {
        theta[0]++;
        tc[0] = 0;
      }
    } else if (a < theta[cpu]) {
      // Decrease theta after enough weak but correct predictions
      tc[0]--;
      if (tc[0] <= -SPEED) {
        theta[0]--;
        tc[0] = 0;
      }
    }
  } 
}
