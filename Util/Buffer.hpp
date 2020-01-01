#ifndef BUFFER_HPP_
#define BUFFER_HPP_

#include <cstdio>
#include <iostream>
#include <vector>

#include <msgpack.hpp>

/**
 * @brief pack data to byte array in MsgPack format
 *
 * this class does not hold any actual buffer data
 */
class Buffer {
  private:
    size_t readPos = 0;
    std::vector<char> *contentPtr = nullptr;

  public:
    /**
     * @brief Construct a new empty Buffer object
     *
     */
    Buffer() {
        readPos = 0;
        contentPtr = nullptr;
    };

    /**
     * @brief Construct a new Buffer object with external buffer
     *
     * buf is empty after this constructor
     *
     * @param buf
     */
    explicit Buffer(std::vector<char> &buf) {
        readPos = 0;
        contentPtr = &buf;
    }

    // copy control
    Buffer(const Buffer &other) = delete;
    Buffer(Buffer &&other) = delete;
    Buffer &operator=(const Buffer &other) = delete;
    Buffer &operator=(Buffer &&other) = delete;

    // destructor
    ~Buffer() = default;

    size_t getReadPos() noexcept { return readPos; }

    void setReadPos(const size_t &pos) noexcept { readPos = pos; }

    /**
     * @brief display readPos and the content pointed by contentPtr
     *
     */
    void dump() noexcept {
        for (auto &v : *contentPtr) {
            printf("%c", v);
        }
        printf("\nreadPos %zu\n", readPos);
    }

    char *getPtr() {
        // can be used to read/write the
        return contentPtr->data();
    }

    size_t getSize() { return contentPtr->size(); }

    /**
     * @brief interface to mimic stringstream as required by msgpack
     *
     * @param ptr
     * @param length
     */
    inline void write(const char *ptr, size_t length) {
        assert(contentPtr != nullptr);
        // for (int i = 0; i < length; i++) {
        //     contentPtr->push_back(*(ptr + i));
        // }
        auto &content = *contentPtr;
        const int currentSize = content.size();
        content.resize(currentSize + length);
        std::copy(ptr, ptr + length, content.end() - length);
    }

    /**
     * @brief pack data into std::vector<char> pointed by contentPtr
     *
     * POD and std data types are supported by msgpack by default
     *
     * @tparam T
     * @param data
     */
    template <class T>
    inline void pack(const T &data) {
        msgpack::pack(*this, data);
    }

    /**
     * @brief unpack the data in content and move the readPos
     *
     * POD and std data types are supported by msgpack by default
     *
     * @tparam T
     * @param output
     * @param content
     */
    template <class T>
    inline void unpack(T &output, const std::vector<char> &content) {
        size_t offset = readPos;
        msgpack::object_handle oh = msgpack::unpack(content.data(), content.size(), offset);
        msgpack::object obj = oh.get();
        obj.convert(output);
        // shift position
        readPos = offset;
        // unpackDebugPrint(obj);
    }

    /**
     * @brief print debug info of msgPack of unpack
     *
     * @param obj
     */
    inline void unpackDebugPrint(const msgpack::object &obj) const {
#ifndef NDEBUG
        // print the deserialized object.
        std::cout << obj << std::endl;
        if (contentPtr != nullptr && readPos > contentPtr->size()) {
            printf("Error: read position past the end of content.\n");
            std::exit(1);
        }
#endif
    }
};

#endif
