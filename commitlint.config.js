module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
      'type-enum': [
        2,
        'always',
        [
          'feat',
          'fix',
          'docs',
          'style',
          'refactor',
          'perf',
          'test',
          'chore'
        ],
      ],
    },
    prompt: {
      messages: {
        skip: ':skip',
        max: 'upper %d chars',
        min: 'lower %d chars',
        emptyWarning: 'không được để trống!',
        upperLimitWarning: 'quá dài',
        lowerLimitWarning: 'quá ngắn'
      },
    },
  };
  