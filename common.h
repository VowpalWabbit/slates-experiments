#pragma once

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <fstream>


enum class platform : int
{
  Mac = 0,
  Windows = 1
};

enum class connection : int
{
  wifi = 0,
  wired = 1
};

enum class region : int
{
  US = 0,
  CA = 1
};


std::string to_string(platform p)
{
  return p == platform::Mac ? "Mac" : "Windows";
}
std::string to_string(connection c)
{
  return c == connection::wifi ? "wifi" : "wired";
}
std::string to_string(region r)
{
  return r == region::US ? "US" : "CA";
}

platform platform_from_string(std::string const& str)
{
  if (str == "0" || str == "mac" || str == "Mac")
  {
    return platform::Mac;
  }
  else if (str == "1" || str == "windows" || str == "Windows")
  {
    return platform::Windows;
  }

  throw new std::runtime_error(str + " is an unknown platform value");
}

connection connection_from_string(std::string const& str)
{
  if (str == "0" || str == "wifi" || str == "Wifi")
  {
    return connection::wifi;
  }
  else if (str == "1" || str == "wired" || str == "Wired")
  {
    return connection::wired;
  }

  throw new std::runtime_error(str + " is an unknown connection value");
}

region region_from_string(std::string const& str)
{
  if (str == "0" || str == "us" || str == "US" || str == "Us")
  {
    return region::US;
  }
  else if (str == "1" || str == "ca" || str == "CA" || str == "Ca")
  {
    return region::CA;
  }

  throw new std::runtime_error(str + " is an unknown region value");
}

using context_t = std::tuple<platform, connection, region>;
using actions_t = std::tuple<std::string, std::string, std::string>;
using record_t = std::tuple<context_t, actions_t, float>;

record_t read_line(std::string const& line)
{
  std::vector<std::string> tokens;
  std::stringstream ss(line);

  std::string intermediate;
  // Tokenize line by comma.
  while (std::getline(ss, intermediate, ','))
  {
    tokens.push_back(intermediate);
  }

  return record_t{
      context_t{platform_from_string(tokens[0]), connection_from_string(tokens[1]), region_from_string(tokens[2])},
      actions_t{tokens[3], tokens[4], tokens[5]}, std::stof(tokens[6])};
}

bool file_exists(std::string const& str)
{
  std::ifstream fs(str);
  return fs.is_open();
}

void read_file(std::string const& file_name, std::map<std::pair<context_t, actions_t>, std::vector<float>>& mapping)
{
  if (!file_exists(file_name))
    throw std::runtime_error(file_name + " not found");
  std::ifstream infile(file_name);

  std::string line;
  // Read and throw away the csv header line.
  std::getline(infile, line);

  while (std::getline(infile, line))
  {
    auto record = read_line(line);
    mapping[{std::get<0>(record), std::get<1>(record)}].push_back(std::get<2>(record));
  }
}

template <size_t index>
std::vector<std::string> get_unique_actions(
    std::map<std::pair<context_t, actions_t>, std::vector<float>> const& mapping)
{
  std::set<std::string> action_set;
  for (auto const& [key, value] : mapping)
  {
    auto const& [context, actions] = key;
    action_set.insert(std::get<index>(actions));
  }

  return std::vector<std::string>(action_set.begin(), action_set.end());
}

context_t generate_random_context()
{
  return context_t{
      static_cast<platform>(rand() % 2), static_cast<connection>(rand() % 2), static_cast<region>(rand() % 2)};
}

float get_reward(const context_t& context, const actions_t& actions,
    std::map<std::pair<context_t, actions_t>, std::vector<float>>& mapping) noexcept
{
  auto it = mapping.find({context, actions});
  if (it != mapping.end())
  {
    return it->second[0];
  }
  return 0.f;
}
