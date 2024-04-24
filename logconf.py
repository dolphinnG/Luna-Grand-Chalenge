import logging
import logging.handlers

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Some libraries attempt to add their own root logger handlers. This is
# annoying and so we get rid of them.
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

logfmt_str = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)03d:%(funcName)s %(message)s"

# formatter = logging.Formatter(logfmt_str)
formatter = logging.Formatter(logfmt_str, datefmt='%Y/%m/%d %H:%M:%S')

streamHandler = logging.StreamHandler() #stderr

streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.DEBUG)

root_logger.addHandler(streamHandler)
"""if you create a new logger with logger_xyz = logging.getLogger('namexyz') 
and don't add any handlers to it, then when you log a message with logger_xyz, 
the message will be passed up to the root logger, 
because the propagation setting is True by default.

The root logger, which has been configured with a handler and a formatter, 
will then handle the message. So in effect, 
logger_xyz will use the same handler and formatter as the root logger,
but this is because the message is being passed up to the root logger,
not because logger_xyz is directly using the root logger's handler and formatter.

If you add a handler to logger_xyz, then logger_xyz will use 
that handler in addition to any handlers in the root logger, 
unless you set logger_xyz.propagate = False, 
in which case it will only use its own handlers."""