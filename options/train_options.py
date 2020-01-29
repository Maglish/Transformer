from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--update_html_freq', type=int, default=15, help='frequency of saving training results to html')


        self.isTrain = True
