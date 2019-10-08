#include <iostream>
#include <memory>
#include <tuple>
#include <array>
#include <algorithm>

#include <live_model.h>
#include <configuration.h>
#include <event_builder.h>
#include <constants.h>

namespace rl = reinforcement_learning;

enum class platform
{
  Mac,
  Windows
};

enum class connection
{
  wifi,
  wired
};

enum class region
{
  US,
  CA
};

struct context_t
{
  platform _platform;
  connection _connection;
  region _region;
};

constexpr float MAX_X = 4.f;
constexpr float MAX_Y = 3.f;
constexpr float MAX_Z = 2.f;
constexpr float RANGE_MIN = 0.f;
constexpr float RANGE_MAX = 1.f;
constexpr float ACTION_STEP_SIZE = 1.f;

constexpr bool operator==(const context_t &a, const context_t &b) noexcept
{
  return std::tie(a._platform, a._connection, a._region) == std::tie(b._platform, b._connection, b._region);
}

constexpr std::array<float, 6> get_coefficients(const context_t &context) noexcept
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

constexpr float get_raw_cost(const context_t &context, float x, float y, float z) noexcept
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

constexpr float rescale_cost(float reward, float values_max, float range_min, float range_max) noexcept
{
  return reward / values_max * (range_max - range_min) + range_min;
}

constexpr float get_reward(const context_t &context, float x, float y, float z) noexcept
{
  const auto raw_reward = get_raw_cost(context, x, y, z);
  const auto max_value = get_raw_cost(context, MAX_X, MAX_Y, MAX_Z);
  return -1 * rescale_cost(raw_reward, max_value, RANGE_MIN, RANGE_MAX);
}

context_t generate_random_context()
{
  return context_t{
      static_cast<platform>(rand() % 2),
      static_cast<connection>(rand() % 2),
      static_cast<region>(rand() % 2)};
}

std::vector<feature_space> generate_actions(float max_value)
{
  std::vector<feature_space> actions(static_cast<int>(max_value / ACTION_STEP_SIZE));
  auto generator = 0.f;
  std::generate(actions.begin(), actions.end(), [&generator]() {
    auto val = generator;
    generator += ACTION_STEP_SIZE;
    return feature_space({{"TAction", {{"value=" + std::to_string(val)}}}});
  });
  return actions;
}

feature_space generate_shared_context(const context_t &context)
{
  feature_space shared_context;
  feat_namespace shared_ns("TShared");
  shared_ns.push_feature("platform=" + std::to_string(static_cast<unsigned int>(context._platform)));
  shared_ns.push_feature("connection=" + std::to_string(static_cast<unsigned int>(context._connection)));
  shared_ns.push_feature("region=" + std::to_string(static_cast<unsigned int>(context._region)));
  shared_context.push_namespace(shared_ns);
  return shared_context;
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
  config.set(rl::name::MODEL_SRC, rl::value::NO_MODEL_DATA);

  auto const err_fn = [](const rl::api_status& status, void*){
    std::cout << status.get_error_msg() << "\n";
  };

  std::unique_ptr<rl::live_model> rl = std::unique_ptr<rl::live_model>(new rl::live_model(config, err_fn));
  if (rl->init(&status) != rl::error_code::success)
  {
    std::cout << status.get_error_msg() << "\n";
    return -1;
  }

  estimator est(rl.get());

  feature_space slate_context;
  feat_namespace shared_ns("Slate");
  shared_ns.push_feature("c");
  slate_context.push_namespace(shared_ns);

  auto x_actions = generate_actions(MAX_X);
  auto y_actions = generate_actions(MAX_Y);
  auto z_actions = generate_actions(MAX_Z);

  problem_data<slate_problem_type> prob_data;
  prob_data.push_slot({slate_context, x_actions});
  prob_data.push_slot({slate_context, y_actions});
  prob_data.push_slot({slate_context, z_actions});

  constexpr auto num_iterations = 10;
  for (auto i = 0; i < num_iterations; i++)
  {
    auto context = generate_random_context();
    auto shared_context = generate_shared_context(context);

    problem_event<slate_problem_type> prob;
    prob.shared_context(shared_context);
    prob.event_problem(prob_data);

    auto ret = est.predict_and_log(prob);
    auto it = ret.begin();
    size_t chosen;

    it->get_chosen_action_id(chosen);
    float x = chosen * ACTION_STEP_SIZE;
    it++;

    it->get_chosen_action_id(chosen);
    float y = chosen * ACTION_STEP_SIZE;
    it++;

    it->get_chosen_action_id(chosen);
    float z = chosen * ACTION_STEP_SIZE;
    it++;

    auto reward = get_reward(context, x, y, z);
    for (auto &item : ret)
    {
      auto event_id = item.get_event_id();
      if (rl->report_outcome(event_id, reward, &status) != rl::error_code::success)
      {
        std::cout << status.get_error_msg() << "\n";
        return -1;
      }
    }
  }
}
