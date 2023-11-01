import platform
import click

from hamacho.core.utils.general import get_cpu_count


class ImageSizeRange(click.ParamType):
    name = "integer"
    default = None
    divisor = 32
    lower_limit = 64
    upper_limit = 640

    def convert(self, value: str, param, ctx) -> int:
        if not value.isdigit():
            click.secho(
                f"WARNING: Training image size must be a positive integer and divisible by {self.divisor}.\n"
                f'"{value}" is not a valid integer. Using default value of model config for training',
                fg="yellow",
            )
            return self.default

        value = int(value)
        new_value = value

        if self.lower_limit > value:
            new_value = self.lower_limit
            click.secho(
                f"WARNING: Training image size must be greater than {self.lower_limit}. "
                f"Adjusting image size to {new_value} from {value}",
                fg="yellow",
            )
        elif value > self.upper_limit:
            new_value = self.upper_limit
            click.secho(
                f"WARNING: Training image size must be smaller than {self.upper_limit}. "
                f"Adjusting image size to {new_value} from {value}",
                fg="yellow",
            )
        elif (value % self.divisor) != 0:
            quotient = value / self.divisor
            new_value = round(quotient) * self.divisor
            click.secho(
                f"WARNING: Training image size must be divisible by {self.divisor}. "
                f"Adjusting image size to {new_value} from {value}",
                fg="yellow",
            )

        return new_value


class BatchSizeType(click.ParamType):
    name = "integer"
    default = None

    def convert(self, value: str, param, ctx) -> int:
        if not value.isdigit():
            click.secho(
                f"WARNING: Training batch size must be a positive integer.\n"
                f'"{value}" is not a valid integer. Using default value of model config for training',
                fg="yellow",
            )
            value = self.default
            return value

        value = int(value)
        if value <= 0:
            click.secho(
                f"WARNING: Training batch size must be a positive integer.\n"
                f'"{value}" doest not fit the condition. Using default value of model config for training',
                fg="yellow",
            )
            value = self.default

        if value != self.default:
            click.secho(f"INFO: batch size set to {value}")

        return value


class TrainTestSplitType(click.ParamType):
    name = "float"
    default = 0.2
    min = 0.01
    max = 0.99

    def convert(self, value: str, param, ctx) -> float:
        try:
            value = float(value)
        except ValueError:
            click.secho(
                f"WARNING: Train Test split percentage must be a float and between {self.min} and {self.max}.\n"
                f'"{value}" is not a valid float. Using default value {self.default} for splitting',
                fg="yellow",
            )
            value = self.default

        if self.min > value or value > self.max:
            click.secho(
                f"WARNING: Train Test split percentage must be between {self.min} and {self.max}.\n"
                f'"{value}" does not fit the condition. Using default value {self.default} for splitting',
                fg="yellow",
            )
            value = self.default

        return value


class NumWorkersType(click.ParamType):
    current_system = platform.system().lower()
    supported_system = "linux"
    name = "integer"
    default, lower_limit = 0, 0
    cpu_headroom = 2
    cpu_count = get_cpu_count()
    upper_limit = cpu_count - cpu_headroom

    def convert(self, value: str, param, ctx) -> int:
        if value != self.default and self.current_system != self.supported_system:
            raise click.NoSuchOption("--num-workers", f"No such option: --num-workers")

        try:
            value = int(value)
        except ValueError:
            click.secho(
                f"WARNING: Number of workers must be an Integer and >= {self.default}.\n"
                f"Setting default value {self.default} for num workers",
                fg="yellow",
            )
            return self.default

        new_value = value
        if value < self.lower_limit:
            new_value = self.lower_limit
            click.secho(
                f"WARNING: Number of workers must be greater or equal to {self.lower_limit}.\n"
                f"Setting num workers to {new_value}",
                fg="yellow",
            )
        elif value > self.cpu_count:
            new_value = self.upper_limit
            click.secho(
                f"WARNING: Given num workers value is more than the actual cpu count ({self.cpu_count}).\n"
                f"Adjusting it to {self.upper_limit}.",
                fg="yellow",
            )
        elif value > self.upper_limit:
            new_value = self.upper_limit
            click.secho(
                f"WARNING: Given num workers value is ({value}) and the actual cpu count is ({self.cpu_count}).\n"
                f"It is advised to keep {self.cpu_headroom} cores free. Adjusting it to {self.upper_limit}.",
                fg="yellow",
            )

        return new_value
