import sys
import logging



class Logger:

    logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
    handler = logging.getLogger("efs-assembler")

    @classmethod
    def time_taken(self, formatted_time_string):
        log = "Feature selection step time taken: " + formatted_time_string
        Logger.handler.info(log)
        return

    

    