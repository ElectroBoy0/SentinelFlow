#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <regex>
#include <algorithm>

namespace py = pybind11;

std::string clean_text(const std::string& input) {
    std::string result = input;
    
    // Convert to lowercase
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
                   
    // Remove HTML tags
    std::regex html_regex("<[^>]*>");
    result = std::regex_replace(result, html_regex, " ");
    
    // Remove URLs
    std::regex url_regex("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+");
    result = std::regex_replace(result, url_regex, " ");
    
    // Remove punctuation and special characters (keep only alphanumeric and spaces)
    std::regex punct_regex("[^a-z0-9\\s]");
    result = std::regex_replace(result, punct_regex, " ");
    
    // Remove multiple spaces
    std::regex space_regex("\\s+");
    result = std::regex_replace(result, space_regex, " ");
    
    // Trim leading/trailing whitespace
    result.erase(0, result.find_first_not_of(" "));
    result.erase(result.find_last_not_of(" ") + 1);
    
    return result;
}

PYBIND11_MODULE(cleaner, m) {
    m.doc() = "High-performance C++ text cleaner plugin"; // optional module docstring
    m.def("clean_text", &clean_text, "A function that cleans text (lowercasing, html, URLs, punctuation)");
}
