#ifndef MSGBUFFER_HPP_
#define MSGBUFFER_HPP_

#include <msgpack.hpp>

#include <cstdio>
#include <iostream>
#include <type_traits>
#include <vector>

/**
 * pack/unpack data in byte array in MsgPack format
 * objects are continuously packced into the content of MsgBuffer class
 * For trivially copyable types, direct memcpy is used. UserType (T) does not need to define the pack() unpack()
 * interface
 * For other types, UserType (T) should define pack(MsgBuffer &) and unpack(MsgBuffer &) when unpacking, there
 * is no need to know beforehand how many bytes are used for each object thanks to the MsgPack format.
 */

class MsgBuffer {
  private:
    size_t readPos = 0;
    std::vector<char> content;

  public:
    // constructor
    explicit MsgBuffer() = default;

    // copy control
    MsgBuffer(const MsgBuffer &other) = default;
    MsgBuffer(MsgBuffer &&other) = default;
    MsgBuffer &operator=(const MsgBuffer &other) = default;
    MsgBuffer &operator=(MsgBuffer &&other) = default;

    // destructor
    ~MsgBuffer() = default;

    size_t getReadPos() noexcept { return readPos; }
    void setReadPos(const size_t &pos) noexcept { readPos = pos; }

    void dump() noexcept {
        printf("\nreadPos %zu\n", readPos);
        // printf("\ncontent data %c:\n", readPos);
        for (auto &v : content) {
            printf("%c", v);
        }
    }

    int getRawSize() { return content.size(); }
    char *getRawPtr() { return content.data(); }

    /**
     * pack() and unpack() for std types internally supported by msgPack
     */
    template <class T>
    inline void pack(const T &data) {
        msgpack::pack(*this, data);
    }

    template <class T>
    inline void unpack(T &output) {
        auto offset = readPos;
        msgpack::object_handle oh = msgpack::unpack(content.data(), content.size(), offset);
        msgpack::object obj = oh.get();
        obj.convert(output);
        readPos = offset; // shift position
        // unpackDebugPrint(obj);
    }

    // helper of deserialization
    inline void unpackDebugPrint(const msgpack::object &obj) const {
        // print the deserialized object.
        std::cout << obj << std::endl;
        if (readPos > content.size()) {
            printf("Error: read position past the end of content.\n");
            exit(1);
        }
    }

    /**
     * pack/unpack non trivial classes, class should define pack() unpack() functions
     */
    template <class T>
    inline void packObj(T &data) {
        data.pack(*this);
    }

    template <class T>
    inline void unpackObj(T &data) {
        data.unpack(*this);
    }

    /**
     * pack/unpack trivially copyable classes, class does not need pack() unpack() functions
     */
    template <class T>
    inline void packTrivCopyable(T &data) {
        static_assert(std::is_trivially_copyable<T>::value);
        const int charBytes = sizeof(T);
        const int contentSize = content.size();
        content.resize(contentSize + charBytes);
        std::memcpy(&(content[contentSize]), &data, charBytes);
    }

    template <class T>
    inline void unpackTrivCopyable(T &data) {
        static_assert(std::is_trivially_copyable<T>::value);
        const int charBytes = sizeof(T);
        const int contentSize = content.size();
        std::memcpy(&data, &(content[readPos]), charBytes);
        readPos += charBytes;
    }

    // interface to mimic stringstream, for MsgPack use
    inline void write(const char *ptr, size_t length) {
        const int currentSize = content.size();
        content.resize(currentSize + length);
        std::copy(ptr, ptr + length, content.end() - length);
    }

    inline void concatenate(const MsgBuffer &other) {
        const int size = content.size();
        const int othersize = other.content.size();
        content.resize(size + othersize);
        std::copy(other.content.cbegin(), other.content.cend(), content.begin() + size);
    }
};

#endif
