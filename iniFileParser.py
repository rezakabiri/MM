from configparser import ConfigParser


class iniFileParser():

    def __init__(self, path):
        self.test_parser = ConfigParser()
        self.test_parser.optionxform = str
        self. test_parser.read(path)

    def get_section_option(self, section, option=None):
        sec_items = self.test_parser.options(section)
        sec_dic = {}
        for item in sec_items:
            sec_dic[item] = self.test_parser.get(section, item)

        if option is not None:
            return sec_dic[option]
        else:
            return sec_dic
