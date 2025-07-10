#pragma once

#include "../core/vector.hpp"
#include "../core/body.hpp"
#include <string>
#include <vector>
#include <memory>
#include <fstream>

namespace orbital_mechanics::io {

/**
 * @brief Data point for simulation output
 */
struct DataPoint {
    double time;
    std::vector<core::Vector2d> positions;
    std::vector<core::Vector2d> velocities;
    
    DataPoint(double t = 0.0) : time(t) {}
    
    void add_body_state(const core::Vector2d& position, const core::Vector2d& velocity) {
        positions.push_back(position);
        velocities.push_back(velocity);
    }
};

/**
 * @brief Abstract base class for data writers
 */
class DataWriter {
public:
    virtual ~DataWriter() = default;
    
    /**
     * @brief Opens the output file/stream
     * @param filename Output filename
     * @return true if successful, false otherwise
     */
    virtual bool open(const std::string& filename) = 0;
    
    /**
     * @brief Writes a data point to the output
     * @param data Data point to write
     * @return true if successful, false otherwise
     */
    virtual bool write(const DataPoint& data) = 0;
    
    /**
     * @brief Closes the output file/stream
     */
    virtual void close() = 0;
    
    /**
     * @brief Gets the format name
     * @return String name of the format
     */
    virtual std::string format_name() const = 0;
};

/**
 * @brief Text file data writer
 * 
 * Writes simulation data to plain text files in columnar format.
 * Each line contains: time, body1_x, body1_y, body1_vx, body1_vy, ...
 */
class TextDataWriter : public DataWriter {
public:
    explicit TextDataWriter(bool write_velocities = false);
    
    bool open(const std::string& filename) override;
    bool write(const DataPoint& data) override;
    void close() override;
    std::string format_name() const override { return "Text"; }
    
    /**
     * @brief Sets the precision for floating point output
     * @param precision Number of decimal places
     */
    void set_precision(int precision) { precision_ = precision; }
    
    /**
     * @brief Sets custom column headers
     * @param headers Vector of column header names
     */
    void set_headers(const std::vector<std::string>& headers);
    
private:
    std::ofstream file_;
    int precision_;
    bool write_velocities_;
    bool headers_written_;
    std::vector<std::string> custom_headers_;
    
    void write_header();
};

/**
 * @brief CSV data writer
 * 
 * Writes simulation data to CSV format files.
 */
class CSVDataWriter : public DataWriter {
public:
    explicit CSVDataWriter(bool write_velocities = false);
    
    bool open(const std::string& filename) override;
    bool write(const DataPoint& data) override;
    void close() override;
    std::string format_name() const override { return "CSV"; }
    
    void set_headers(const std::vector<std::string>& headers);
    
private:
    std::ofstream file_;
    bool write_velocities_;
    bool headers_written_;
    std::vector<std::string> custom_headers_;
    
    void write_header();
};

#ifdef USE_HDF5
/**
 * @brief HDF5 data writer
 * 
 * Writes simulation data to HDF5 format for efficient storage
 * and analysis with scientific computing tools.
 */
class HDF5DataWriter : public DataWriter {
public:
    explicit HDF5DataWriter(bool write_velocities = false);
    ~HDF5DataWriter();
    
    bool open(const std::string& filename) override;
    bool write(const DataPoint& data) override;
    void close() override;
    std::string format_name() const override { return "HDF5"; }
    
    /**
     * @brief Sets compression level (0-9)
     * @param level Compression level (0 = no compression, 9 = maximum)
     */
    void set_compression_level(int level) { compression_level_ = level; }
    
private:
    struct HDF5Impl;
    std::unique_ptr<HDF5Impl> impl_;
    bool write_velocities_;
    int compression_level_;
};
#endif

/**
 * @brief Factory function to create data writers
 * @param format Format type ("text", "csv", "hdf5")
 * @param write_velocities Whether to include velocity data
 * @return Unique pointer to the created writer
 */
std::unique_ptr<DataWriter> create_data_writer(const std::string& format, 
                                              bool write_velocities = false);

}  // namespace orbital_mechanics::io