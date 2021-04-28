from datetime import date, datetime

class DateTimeOperator:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_today():
        return date.today().strftime("%d_%m_%Y")

    @staticmethod
    def get_current_date_and_time():
        return datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

#print(DateTimeOperator.get_current_date_and_time())
#print(DateTimeOperator.get_today())