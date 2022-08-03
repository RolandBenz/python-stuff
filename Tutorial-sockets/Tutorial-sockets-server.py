"""
Okay, so what is a socket?
The socket itself is just one of the endpoints in a communication between programs on some network.
A socket will be tied to some port on some host.
In general, you will have either a client or a server type of entity or program.
In the case of the server, you will bind a socket to some port on the server (localhost).
In the case of a client, you will connect a socket to that server,
on the same port that the server-side code is using."""


# to run:
# 1. open terminal: Tutorial-sockets>python Tutorial-sockets-server.py
# 2. open terminal: Tutorial-sockets>python Tutorial-sockets-client.py


import socket
import time
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
# 2. For IP sockets, the address that we bind to is
# a tuple of the hostname and the port number.
"""
s.bind((socket.gethostname(), 54276))

"""
# 3. Let's make a queue of 5
# We can only handle one connection at a given time, so we want to allow for some sort of queue,
# just in case we get a slight burst.
# If someone attempts to connect while the queue is full, they will be denied.
# Let's make a queue of 5:
"""
s.listen(5)

"""
# 4. And now, we just listen!
"""
while True:
    """
    # 4.1 now our endpoint knows about the OTHER endpoint.
    """
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")

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
        """
        # let's convert a dict to a byte string with pickle
        """
        d = {1: "hi", 2: "there"}
        msg = pickle.dumps(d)

        """
        # we're sending a header that counts in bytes, rather than characters.
        """
        msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
        print(msg)

        """
        # 4.2 Our sockets can send and recv data.
        """
        clientsocket.send(msg)

    else:
        """
        # 4.2 Our sockets can send and recv data. These methods of handling data deal in buffers.
        # Buffers happen in chunks of data of some fixed size. Let's see that in action:
        """
        #clientsocket.send(bytes("Hey there!!!","utf-8"))

        """
        # So now our messages will have a header of HEADERSIZE = 10 characters/bytes 
        # that will contain the length of the message, which our client use to inform it 
        # when the end of the message is received.
        """
        msg_ = "Welcome to the server!"
        msg = f"{len(msg_):<{HEADERSIZE}}" + msg_
        clientsocket.send(bytes(msg, "utf-8"))

        """
        # Let's do an example where the server just streams out something simple, like the current time.
        """
        while True:
            time.sleep(10)
            msg_ = f"The time is {time.time()}"
            msg = f"{len(msg_):<{HEADERSIZE}}" + msg_
            print(msg)
            clientsocket.send(bytes(msg, "utf-8"))

        """
        # This connection now is remaining open. 
        # This is due to our while loop in the client socket. 
        # We can use .close() on a socket to close it if we wish. 
        # We can do this either on the server, or on the client...or both. 
        # It's probably a good idea to be prepared for either connection to drop or be closed for whatever reason.
        """
        clientsocket.close()

