#include <iostream>
#include <memory>
#include <tuple>
#include <array>
#include <algorithm>
#include <sstream>

#include <live_model.h>
#include <configuration.h>
#include <event_builder.h>
#include <constants.h>
#include <fstream>
#include <utility>
#include <map>
#include <set>
#include <unordered_set>

#include "common.h"

namespace rl = reinforcement_learning;

constexpr float MAX_X = 4.f;
constexpr float MAX_Y = 3.f;
constexpr float MAX_Z = 2.f;
constexpr float RANGE_MIN = 0.f;
constexpr float RANGE_MAX = 1.f;
constexpr float ACTION_STEP_SIZE = 0.1f;

constexpr std::array<float, 6> get_coefficients(const context_t& context) noexcept
{
  if (context == context_t{platform::Mac, connection::wifi, region::US})
    return {0.37755595, 0.88794085, 0.78759054, 0.60708194, 0.92570716, 0.35602915};
  else if (context == context_t{platform::Mac, connection::wifi, region::CA})
    return {0.19113076, 0.17363948, 0.64172931, 0.5095073, 0.45841506, 0.43078203};
  else if (context == context_t{platform::Mac, connection::wired, region::US})
    return {0.78266802, 0.2267633, 0.95940249, 0.33171948, 0.36201023, 0.36354627};
  else if (context == context_t{platform::Mac, connection::wired, region::CA})
    return {0.55245693, 0.95071475, 0.21295371, 0.35589226, 0.25239824, 0.6135975};
  else if (context == context_t{platform::Windows, connection::wifi, region::US})
    return {0.14937779, 0.46339031, 0.49216011, 0.14819576, 0.47126218, 0.26317773};
  else if (context == context_t{platform::Windows, connection::wifi, region::CA})
    return {0.14937779, 0.46339031, 0.49216011, 0.14819576, 0.47126218, 0.26317773};
  else if (context == context_t{platform::Windows, connection::wired, region::US})
    return {0.14937779, 0.46339031, 0.49216011, 0.14819576, 0.47126218, 0.26317773};
  else if (context == context_t{platform::Windows, connection::wired, region::CA})
    return {0.87995436, 0.10853902, 0.24386487, 0.14241173, 0.30777027, 0.14954826};
}

constexpr float get_raw_reward(const context_t& context, float x, float y, float z) noexcept
{
  const auto coefficients = get_coefficients(context);
  auto reward = 0.f;
  reward += coefficients[0] * x;
  reward += coefficients[1] * y;
  reward += coefficients[2] * z;
  reward += coefficients[3] * x * y;
  reward += coefficients[4] * x * z;
  reward += coefficients[5] * y * z;
  return reward;
}

constexpr float rescale_reward(float reward, float values_max, float range_min, float range_max) noexcept
{
  return reward / values_max * (range_max - range_min) + range_min;
}


float get_reward(const context_t& context, float x, float y, float z) noexcept
{
  const auto raw_cost = get_raw_reward(context, x, y, z);
  const auto max_value = get_raw_reward(context, MAX_X, MAX_Y, MAX_Z);
  return rescale_reward(raw_cost, max_value, RANGE_MIN, RANGE_MAX);
}

std::vector<feature_space> generate_actions(float max_value)
{
  std::vector<feature_space> actions(static_cast<int>(max_value / ACTION_STEP_SIZE));
  auto generator = 0.f;
  std::generate(actions.begin(), actions.end(), [&generator]() {
    auto val = generator;
    generator += ACTION_STEP_SIZE;
    return feature_space({{"Action", {{"value=" + std::to_string(val)}}}});
  });
  return actions;
}

std::vector<feature_space> generate_actions(std::vector<std::string> stractions)
{
  std::vector<feature_space> actions;
  actions.reserve(stractions.size());
  for (auto const& a : stractions)
  {
    actions.push_back(feature_space({{"Action", {{"value=" + a}}}}));
  }
  return actions;
}

feature_space generate_shared_context(const context_t& context)
{
  feature_space shared_context;
  feat_namespace shared_ns("User");
  shared_ns.push_feature("platform=" + std::to_string(static_cast<int>(std::get<0>(context))));
  shared_ns.push_feature("connection=" + std::to_string(static_cast<int>(std::get<1>(context))));
  shared_ns.push_feature("region=" + std::to_string(static_cast<int>(std::get<2>(context))));
  shared_context.push_namespace(shared_ns);
  return shared_context;
}

void average_rewards_and_output(std::vector<std::string> const& files)
{
  std::map<std::pair<context_t, actions_t>, std::vector<float>> mapping;

  for (auto const& file : files)
  {
    read_file(file, mapping);
  }

  std::cout << "platform,network,country,x,y,z,reward\n";
  for (auto const& [key, value] : mapping)
  {
    auto const& [context, actions] = key;
    std::cout << to_string(std::get<0>(context)) << "," << to_string(std::get<1>(context)) << ","
              << to_string(std::get<2>(context)) << "," << std::get<0>(actions) << "," << std::get<1>(actions)
              << "," << std::get<2>(actions) << "," << std::accumulate(value.begin(), value.end(), 0.0) / value.size()
              << "\n";
  }
}

void find_minimums(std::map<std::pair<context_t, actions_t>, std::vector<float>>& mapping)
{
  auto x_actions = get_unique_actions<0>(mapping);
  auto y_actions = get_unique_actions<1>(mapping);
  auto z_actions = get_unique_actions<2>(mapping);

  std::cout << "platform,network,country,x,y,z,min_reward\n";
  for (int plat = 0; plat < 2; plat++)
  {
    for (int conn = 0; conn < 2; conn++)
    {
      for (int reg = 0; reg < 2; reg++)
      {
        float min = 10000.f;
        std::string x_min;
        std::string y_min;
        std::string z_min;
        for (auto& x : x_actions)
        {
          for (auto& y : y_actions)
          {
            for (auto& z : z_actions)
            {
              auto current = mapping[{
                  context_t{static_cast<platform>(plat), static_cast<connection>(conn), static_cast<region>(reg)},
                  actions_t{x, y, z}}][0];
              if(current < min)
              {
                min = current;
                x_min = x;
                y_min = y;
                z_min = z;
              }
            }
          }
        }
         std::cout << to_string(static_cast<platform>(plat)) << "," << to_string(static_cast<connection>(conn)) << ","
              << to_string(static_cast<region>(reg)) << "," << x_min << "," << y_min
              << "," << z_min << "," << min
              << "\n";
      }
    }
  }
}

void find_average(std::map<std::pair<context_t, actions_t>, std::vector<float>>& mapping)
{
  auto x_actions = get_unique_actions<0>(mapping);
  auto y_actions = get_unique_actions<1>(mapping);
  auto z_actions = get_unique_actions<2>(mapping);

  std::cout << "platform,network,country,x,y,z,min_reward\n";
  for (int plat = 0; plat < 2; plat++)
  {
    for (int conn = 0; conn < 2; conn++)
    {
      for (int reg = 0; reg < 2; reg++)
      {
        float avg = 0.f;
        int num = 0;
        for (auto& x : x_actions)
        {
          for (auto& y : y_actions)
          {
            for (auto& z : z_actions)
            {
              auto current = mapping[{
                  context_t{static_cast<platform>(plat), static_cast<connection>(conn), static_cast<region>(reg)},
                  actions_t{x, y, z}}][0];
              avg += current;
              num++;
            }
          }
        }
         std::cout << plat << "," << conn << ","
              << reg << "," << avg / num
              << "\n";
      }
    }
  }
}

int main()
{
  rl::api_status status;
  rl::utility::configuration config;
  config.set(rl::name::APP_ID, "slate_simulator_test");
  config.set(rl::name::OBSERVATION_SENDER_IMPLEMENTATION, rl::value::OBSERVATION_FILE_SENDER);
  config.set(rl::name::INTERACTION_SENDER_IMPLEMENTATION, rl::value::INTERACTION_FILE_SENDER);
  config.set(rl::name::DECISION_SENDER_IMPLEMENTATION, rl::value::INTERACTION_FILE_SENDER);
  config.set(rl::name::INITIAL_EPSILON, "1.0");
  config.set(rl::name::MODEL_SRC, rl::value::FILE_MODEL_DATA);
  config.set(rl::name::MODEL_FILE_NAME, "./input.model");
  config.set(rl::name::CCB_SAMPLE_MODE, rl::value::SAMPLE_SINGLE);
  config.set(
      rl::name::MODEL_VW_INITIAL_COMMAND_LINE, "--ccb_explore_adf --slate --json --quiet --epsilon 1.0 --id N/A");

  // Uncomment this to produced average reward mapping.
  // average_rewards_and_output({"../data/fine/simulation_data_['Mac', 'wifi', 'CA'].csv",
  //     "../data/fine/simulation_data_['Mac', 'wifi', 'US'].csv", "../data/fine/simulation_data_['Mac', 'wired', 'CA'].csv",
  //     "../data/fine/simulation_data_['Mac', 'wired', 'US'].csv", "../data/fine/simulation_data_['Windows', 'wifi', 'CA'].csv",
  //     "../data/fine/simulation_data_['Windows', 'wifi', 'US'].csv", "../data/fine/simulation_data_['Windows', 'wired', 'CA'].csv",
  //     "../data/fine/simulation_data_['Windows', 'wired', 'US'].csv"});
  // return 0;

  std::map<std::pair<context_t, actions_t>, std::vector<float>> mapping;
  read_file("/mnt/c/w/repos/slate_sim/data/coarse/averaged.csv", mapping);

  // find_minimums(mapping);
  // // find_average(mapping);
  // return 0;

  auto const err_fn = [](const rl::api_status& status, void*) { std::cout << status.get_error_msg() << "\n"; };

  float total_reward = 0.f;
  size_t num_rewards = 0;
  std::unique_ptr<rl::live_model> rl = std::unique_ptr<rl::live_model>(new rl::live_model(config, err_fn));
  if (rl->init(&status) != rl::error_code::success)
  {
    std::cout << status.get_error_msg() << "\n";
    return -1;
  }

  estimator est(rl.get());

  feature_space slate_context;
  feat_namespace shared_ns("Slate");
  shared_ns.push_feature("constant");
  slate_context.push_namespace(shared_ns);

  auto x_actions = get_unique_actions<0>(mapping);
  auto y_actions = get_unique_actions<1>(mapping);
  auto z_actions = get_unique_actions<2>(mapping);

  auto x_feat_spaces = generate_actions(x_actions);
  auto y_feat_spaces = generate_actions(y_actions);
  auto z_feat_spaces = generate_actions(z_actions);

  problem_data<slate_problem_type> prob_data;
  prob_data.push_slot({slate_context, x_feat_spaces});
  prob_data.push_slot({slate_context, y_feat_spaces});
  prob_data.push_slot({slate_context, z_feat_spaces});

  problem_event<slate_problem_type> prob;
  prob.event_problem(prob_data);

  constexpr auto num_iterations = 10000;
  for (auto i = 0; i < num_iterations; i++)
  {
    auto context = generate_random_context();
    auto shared_context = generate_shared_context(context);
    prob.shared_context(shared_context);

    auto ret = est.predict_and_log(prob);
    auto it = ret.begin();
    size_t chosen;

    if (it->get_chosen_action_id(chosen, &status) != rl::error_code::success)
    {
      std::cout << status.get_error_msg() << "\n";
      return -1;
    }
    auto x = x_actions[chosen];
    it++;

    if (it->get_chosen_action_id(chosen, &status) != rl::error_code::success)
    {
      std::cout << status.get_error_msg() << "\n";
      return -1;
    }
    auto y = y_actions[chosen];
    it++;

    if (it->get_chosen_action_id(chosen, &status) != rl::error_code::success)
    {
      std::cout << status.get_error_msg() << "\n";
      return -1;
    }
    auto z = z_actions[chosen];
    it++;

    // The reward is supposed to be minimized so we negate it here.
    auto reward = -1 * get_reward(context, {x, y, z}, mapping);
    total_reward += reward;
    num_rewards++;

    if (i % 500 == 0)
    {
      std::cout << "i: " << i << ", Avg reward: " << total_reward / num_rewards << ", this reward: " << reward << "\n";
    }

    for (auto& item : ret)
    {
      auto event_id = item.get_event_id();
      if (rl->report_outcome(event_id, reward, &status) != rl::error_code::success)
      {
        std::cout << status.get_error_msg() << "\n";
        return -1;
      }
    }
  }
  std::cout << "Total iterations: " << num_iterations << ", Avg reward: " << total_reward / num_rewards << "\n";
}
