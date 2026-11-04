#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

// Minimal "--key value" parser shared by the command-line tools.
class Args {
public:
    Args(int argc, char** argv, std::map<std::string, std::string> defaults, std::string usage)
        : values_(std::move(defaults)), usage_(std::move(usage)) {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "-h" || key == "--help") fail();
            if (key.rfind("--", 0) != 0 || !values_.count(key.substr(2)) || i + 1 >= argc) {
                std::cerr << "unexpected argument: " << key << "\n";
                fail();
            }
            values_[key.substr(2)] = argv[++i];
        }
    }

    const std::string& str(const std::string& key) const { return values_.at(key); }
    int integer(const std::string& key) const { return std::stoi(values_.at(key)); }

private:
    [[noreturn]] void fail() const {
        std::cerr << usage_ << "\n";
        std::exit(2);
    }
    std::map<std::string, std::string> values_;
    std::string usage_;
};
