#pragma once

#include "data_writer.hpp"
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace orbital_mechanics::io {

inline TextDataWriter::TextDataWriter(bool write_velocities)
    : precision_(6), write_velocities_(write_velocities), headers_written_(false) {}

inline bool TextDataWriter::open(const std::string& filename) {
    file_.open(filename);
    if (!file_.is_open()) {
        return false;
    }
    
    file_ << std::fixed << std::setprecision(precision_);
    headers_written_ = false;
    return true;
}

inline bool TextDataWriter::write(const DataPoint& data) {
    if (!file_.is_open()) {
        return false;
    }
    
    if (!headers_written_) {
        write_header();
        headers_written_ = true;
    }
    
    file_ << data.time;
    
    for (size_t i = 0; i < data.positions.size(); ++i) {
        file_ << " " << data.positions[i].x << " " << data.positions[i].y;
        if (write_velocities_ && i < data.velocities.size()) {
            file_ << " " << data.velocities[i].x << " " << data.velocities[i].y;
        }
    }
    
    file_ << "\n";
    return true;
}

inline void TextDataWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

inline void TextDataWriter::set_headers(const std::vector<std::string>& headers) {
    custom_headers_ = headers;
}

inline void TextDataWriter::write_header() {
    if (!custom_headers_.empty()) {
        file_ << "# ";
        for (size_t i = 0; i < custom_headers_.size(); ++i) {
            if (i > 0) file_ << " ";
            file_ << custom_headers_[i];
        }
        file_ << "\n";
    } else {
        file_ << "# Time(s)";
        // We don't know the number of bodies at this point, so we'll write a generic header
        file_ << " [Body positions and velocities follow]\n";
    }
}

inline CSVDataWriter::CSVDataWriter(bool write_velocities)
    : write_velocities_(write_velocities), headers_written_(false) {}

inline bool CSVDataWriter::open(const std::string& filename) {
    file_.open(filename);
    if (!file_.is_open()) {
        return false;
    }
    
    headers_written_ = false;
    return true;
}

inline bool CSVDataWriter::write(const DataPoint& data) {
    if (!file_.is_open()) {
        return false;
    }
    
    if (!headers_written_) {
        write_header();
        headers_written_ = true;
    }
    
    file_ << data.time;
    
    for (size_t i = 0; i < data.positions.size(); ++i) {
        file_ << "," << data.positions[i].x << "," << data.positions[i].y;
        if (write_velocities_ && i < data.velocities.size()) {
            file_ << "," << data.velocities[i].x << "," << data.velocities[i].y;
        }
    }
    
    file_ << "\n";
    return true;
}

inline void CSVDataWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

inline void CSVDataWriter::set_headers(const std::vector<std::string>& headers) {
    custom_headers_ = headers;
}

inline void CSVDataWriter::write_header() {
    if (!custom_headers_.empty()) {
        for (size_t i = 0; i < custom_headers_.size(); ++i) {
            if (i > 0) file_ << ",";
            file_ << custom_headers_[i];
        }
        file_ << "\n";
    } else {
        file_ << "Time";
        // Generic header for unknown number of bodies
        file_ << ",[Body data follows]\n";
    }
}

inline std::unique_ptr<DataWriter> create_data_writer(const std::string& format, 
                                                     bool write_velocities) {
    std::string lower_format = format;
    std::transform(lower_format.begin(), lower_format.end(), lower_format.begin(), ::tolower);
    
    if (lower_format == "text" || lower_format == "txt") {
        return std::make_unique<TextDataWriter>(write_velocities);
    } else if (lower_format == "csv") {
        return std::make_unique<CSVDataWriter>(write_velocities);
    }
#ifdef USE_HDF5
    else if (lower_format == "hdf5" || lower_format == "h5") {
        return std::make_unique<HDF5DataWriter>(write_velocities);
    }
#endif
    else {
        throw std::invalid_argument("Unknown data writer format: " + format);
    }
}

}  // namespace orbital_mechanics::io