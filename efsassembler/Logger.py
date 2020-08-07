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
    
    @classmethod
    def end_experiment_message(self):
        Logger.handler.info("Experiment finished!")
        return

    @classmethod
    def decoding_dataframe(self):
        Logger.handler.info("Decoding dataframe")
        return

    @classmethod
    def starting_evaluation_process(self):
        Logger.handler.info("Starting evaluation process")
        return

    @classmethod
    def creating_csv_files(self):
        Logger.handler.info("Creating csv files")
        return

    @classmethod
    def evaluating_inner_levels(self):
        Logger.handler.info("Evaluating inner levels")
        return
    

    