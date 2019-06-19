from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_dir, comment=None):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        self.comment = comment
        if comment is None:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = SummaryWriter(log_dir, comment=comment)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for pair in tag_value_pairs:
            tag, value = pair
            self.scalar_summary(tag, value, step)