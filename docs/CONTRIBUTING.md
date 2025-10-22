# Contributing

If you are a dev contributing to the Retriever codebase, there are a few things you should know.

## Task System

Retriever makes use of [taskipy](https://github.com/taskipy/taskipy) to create shortcuts for common local workflows. See the [pyproject.toml](../pyproject.toml) file for a list of tasks under `[tool.taskipy.tasks]`.

## Setting up your workspace

Retriever makes use of the following to ensure code quality and readability:

- [Ruff](https://docs.astral.sh/ruff/); To lint/fix code style issues and to format code to a set standard.
- [Basedpyright](https://docs.basedpyright.com/latest/); To enforce static typing and ensure type ambiguities don't make it in and cause runtime errors or later development headaches.

Both of these have already been configured (see [pyproject.toml](../pyproject.toml)) to a relatively strict standard, and run automatically on PRs. Apologies for the strictness, but it's worth it :)

It's recommended you configure your editor of choice to use these, so you can more easily conform to the standards while writing code. Documentation for IDE integration is available for both : [Ruff](https://docs.astral.sh/ruff/editors/setup/), [Basedpyright](https://docs.basedpyright.com/latest/installation/ides/). Integration should be relatively seamless for the common editors (VSCode, Sublime, Neovim, Emacs, etc.).

### Commit Hooks

When you are writing code changes, it is highly recommended to install the project's commit hooks to enforce type checking and good code behavior:

```bash
task hook # auto-installs for this project
```

Commit hooks will run [pre-commit](https://pre-commit.com/), configured to do the following:

- Remove trailing whitespace
- Fix file endings
- Prevent large (MB scale) files from being committed
- Check that the staged code conforms to lint standards, and formats it.
- Check that the staged code passes static type checking
- Check that there are no dependency errors

### Best practices

It's good practice to run `task fixup` periodically or prior to commiting-- this will automatically run linting, formatting, type checking, and dependency checking. With commit hooks enabled, this will avoid most cases of failing to make a commit due to pre-commit complaints.

### Testing

Retriever makes use of [pytest](https://docs.pytest.org/en/stable/) for unit testing, and [CodeCov](https://about.codecov.io/) to check for code coverage. These are run automatically on PRs.

It's recommended to run tests locally as well, so you can gauge the test coverage, and to ensure your changes aren't breaking anything, before you make your PR. You can do this easily by running `task test` to view both test results and coverage.

In addition to running unit tests, it's highly recommended to run Retriever locally so you can integration-test your changes. `task dev` has been included to make it easy to run a simple setup with trace-level debugging. From there, you can send requests to it locally using cURL or your REST tools of choice. The first thing maintainers will do when reviewing your PR (after glancing over the changes) is to run Retriever locally and test a relevant query, so being able to say it works on your system is very useful when you get feedback.

You may also wish to check that your PR still builds a Docker image correctly (usually only relevant if you've added dependencies). To do so, you can run `docker compose build`.

## Writing a PR

When writing a PR, it is recommended to use a descriptive title and try to summarize the changes in a few bullet points.

Upon making the PR, GitHub Actions will run, doing the following things:

- Check that the PR adheres to linting/formatting standards.
- Check that it passes static type checking.
- Check that it passes existing unit tests.
- Check that test coverage is, at worst, not decreased by the PR.
- Check that the PR is capable of building a new Docker image

If any of these fail, the PR will be marked as failing checks. At that point, it's recommended you review the workflow output and make changes as appropriate. The only check that may see some leniency is test coverage, depending on time priorities. Most of these can be tested ahead of time if you set up your workspace according to the above.
