# Contributing to jax-rs
First off, thank you for considering contributing to jax-rs. 
It's people like you that make jax-rs great!


## Getting Started

### 1. Fork the Project
Start by forking the repository and then clone your fork.

```bash
git clone https://github.com/cryptopatrick/jax-rs.git
cd jax-rs
```

### 2. Install Rust
If you haven't already, install Rust by following the instructions [here](https://www.rust-lang.org/tools/install).

### 3a. Install tools
If you haven't already, install Just by following the instrutions [here](https://github.com/casey/just).

Install tools by running `install-required`, `install-recommended` or `install-all'

```bash
just install-all
```

### 3b. Running and Testing

To run jax-rs, piping the output into [bunyan](https://github.com/LukeMathWalker/bunyan) to format outputted the structured logs:

```bash
cargo run | bunyan
```

To run the tests:

```bash
cargo test
```

### 4. Create a Branch
Create a branch for your new feature or bug fix:

```bash
git checkout -b YOUR_USERNAME/your-new-feature
```

## Coding Guidelines
Please adhere to the following guidelines:

- Follow the [Rust Coding Guidelines](https://rust-lang.github.io/api-guidelines/about.html).
- Write tests for new features.
- Keep commits small and focused.
- Update the documentation as needed.

## Submitting a Pull Request
1. Push your changes to your fork:

```bash
git push origin YOUR_USERNAME/your-new-feature
```

2. [Submit a pull request](https://github.com/cryptopatrick/jax-rs/pulls) to the original repository.

3. Please include a clear and descriptive title and a detailed explanation of your changes in the pull request description.

4. Link any relevant issues in your pull request description.

## Reporting Issues
Please use the [issue tracker](https://github.com/cryptopatrick/jax-rs/issues) to report any bugs or request new features.

## Licensing
By contributing, you agree that your contributions will be licensed under the same license as the original project MIT.

## Questions?
Feel free to reach out to the maintainers or open an issue. 
Always happy to help!

Thank you for contributing to jax-rs!
