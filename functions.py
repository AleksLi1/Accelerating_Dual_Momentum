from datetime import datetime, date


def start_of_year(my_date):

    """
    Gives the starting date of a year given any date
    :param my_date: date, str
    :return: str
    """

    my_date = datetime.strptime(my_date, '%Y-%m-%d')
    starting_date = date(my_date.year, my_date.month, 1)
    starting_date = starting_date.strftime('%Y-%m-%d')
    return starting_date
