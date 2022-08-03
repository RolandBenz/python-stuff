"""
Okay, so what is a socket?
The socket itself is just one of the endpoints in a communication between programs on some network.
A socket will be tied to some port on some host.
In general, you will have either a client or a server type of entity or program.
In the case of the server, you will bind a socket to some port on the server (localhost).
In the case of a client, you will connect a socket to that server,
on the same port that the server-side code is using.
"""


# to run:
# 1. open terminal: Tutorial-sockets>python Tutorial-sockets-server.py
# 2. open terminal: Tutorial-sockets>python Tutorial-sockets-client.py


import socket
import pickle


HEADERSIZE = 10
PICKLE = True


"""
# 1. create the socket
# AF_INET == ipv4
# SOCK_STREAM == TCP
"""
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

"""
# 4.1. Now, since this is the client, rather than binding, we are going to connect.
# In the more traditional sense of client and server,
# you wouldn't actually have the client and server on the same machine.
# If you wanted to have two programs talking to each other locally, you could do this,
# but typically, your client will more likely connect to some external server,
# using its public IP address, not socket.gethostname(). You will pass the string of the IP instead.
"""
s.connect((socket.gethostname(), 54276))

"""
# 4.2 Buffer size
# This means our socket is going to attempt to receive data from the server,
# in a buffer size of 1024 bytes at a time.
"""
#msg = s.recv(1024)

"""
# 4.2.1 Then, let's just do something basic with the data we get, like print it out!
"""
#print(msg.decode("utf-8"))

"""
# 4.2 Buffer size and loop
# (At some point, no matter what number you set for the buffer size,
# many applications that use sockets will eventually desire to send some amount of bytes far over the buffer size.
# Instead, we need to probably build our program from the ground up to actually accept
# the entirety of the messages in chunks of the buffer, even if there's usually only one chunk.
# We do this mainly for memory management.)
"""
while True:
    if PICKLE:
        full_msg = b''
    else:
        full_msg = ''

    """
    So, we start off in a state where the next bit of data we get is a new_msg.
    """
    new_msg = True
    msglen = -(HEADERSIZE+1)

    while True:
        """
        # buffer size, e.g 16 means 16 characters
        """
        msg = s.recv(16)
        #print(msg.decode("utf-8"))

        """
        If the message is a new_msg, then the first thing we do is parse the header, 
        which we already know is a fixed-length of 10 characters. 
        From here, we parse the message length.
        """
        if new_msg:
            print("new msg len:", msg[:HEADERSIZE])
            msglen = int(msg[:HEADERSIZE])
            new_msg = False

        """
        If we run this, however, we will see our client.py then spams out a bunch of nothingness, 
        because the data it's receiving, is, well, nothing. It's empty. 
        0 bytes, but we are still asking it to print out what it receives, even if that's nothing! 
        We could fix that, break breaks this while loop:
        """
        if len(msg) <= 0:
            break

        """
        # In Python, everything is an object, and all of your objects can be serialized with Pickle. 
        # Serialization is the conversion of your object to bytes.
        # ...and we send bytes with sockets. 
        # This means that you can communicate between your python programs both locally, 
        # or remotely, via sockets, using pickle. So now, literally anything...
        # functions, a giant dictionary, some arrays, a TensorFlow model...etc 
        # can be sent back and forth between your programs!
        """
        if PICKLE:
            print(f"full message length: {msglen}")
            full_msg += msg
            if len(full_msg) > 0:
                print(len(full_msg))
            if len(full_msg) - HEADERSIZE == msglen:
                print("full msg recvd")
                print(full_msg[HEADERSIZE:])
                print(pickle.loads(full_msg[HEADERSIZE:]))
                new_msg = True
                full_msg = b""
        else:
            """
            So, now we are buffering through the full message. 
            with s.recv(16) the method msg.decode("utf-8") delivers the first 16 bits, then the next 16 bits, etc.
            """
            full_msg += msg.decode("utf-8")
            if len(full_msg) > 0:
                print(full_msg)

            """
            Above, we continue to build the full_msg, 
            until that var is the size of msglen + our HEADERSIZE. 
            Once this happens, we print out the full message.
            """
            if len(full_msg) - HEADERSIZE == msglen:
                print("full msg recvd")
                print(full_msg[HEADERSIZE:])
                new_msg = True
                full_msg = ""