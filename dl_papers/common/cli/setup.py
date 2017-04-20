import logging

import click

__all__ = ('configure_script', 'cli')

# -----------------------------------------------------------------------------


def configure_script():
    # Clear existing handlers in case e.g. a module-level call to logging
    # already set some up, as logging.basicConfig will do nothing if the root
    # logger already has handlers.
    del logging.getLogger().handlers[:]

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        level=logging.INFO,
    )


# -----------------------------------------------------------------------------


@click.group()
def cli():
    configure_script()
