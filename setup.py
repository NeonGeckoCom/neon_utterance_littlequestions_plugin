#!/usr/bin/env python3
from setuptools import setup


PLUGIN_ENTRY_POINT = 'neon_utterance_littlequestions_plugin=neon_utterance_littlequestions_plugin:QuestionTagger'
setup(
    name='neon_utterance_littlequestions_plugin',
    version='0.0.1',
    description='A utterance parser/classifier/transformer plugin for ovos/neon/mycroft',
    url='https://github.com/NeonGeckoCom/neon_utterance_littlequestions_plugin',
    author='JarbasAi',
    author_email='jarbasai@mailfence.com',
    license='bsd3',
    packages=['neon_utterance_littlequestions_plugin'],
    zip_safe=True,
    keywords='mycroft plugin utterance parser/classifier/transformer',
    entry_points={'neon.plugin.text': PLUGIN_ENTRY_POINT}
)
