# torchsvm

This is a torchbased package to solve kernel SVM fastly.

## Table of contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Usage](#usage)
* [Known issues and limitations](#known-issues-and-limitations)
* [Getting help](#getting-help)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)


## Introduction

torchsvm, a PyTorch-based library that trains kernel SVMs and other large-margin classifiers with exact leave-one-out cross-validation (LOOCV) error computation. Conventional SVM solvers often face scalability and efficiency challenges, especially on large datasets or when multiple cross-validation runs are required. TorchKSVM computes LOOCV at the same cost as training a single SVM while boosting speed and scalability via CUDA-accelerated matrix operations. Benchmark experiments indicate that TorchKSVM outperforms existing kernel SVM solvers in efficiency and speed. 
\end{document}

For a real software project, the [Introduction](#introduction) section – which you are presently reading – should summarize the motivations for creating the software, the need(s) addressed by the software, what the software does, and optionally, links to resources that can help orient readers. Ideally, this section should be short and use plain language. Keep in mind that not all readers will be familiar with the topic area.


## Installation

Begin this [Installation](#installation) section by describing prerequisites needed to use the software. Examples include required hardware, operating systems, software frameworks, compilers, and/or interpreters.

Next, provide step-by-step instructions for installing your software, preferably with examples of commands that can be copy-pasted by readers into their computing environments. If your software can be installed using common installers or package managers (e.g., `pip`, `npm`, `brew`, `apt`, etc.), illustrate how it can be done using [code blocks](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks) in the Markdown file so that it's clear to readers. For example,

```sh
pip install yoursoftware
```

For installation methods that don't involve command lines, providing screenshots along with written instructions can help readers figure out what they need to do.

Subsections may be appropriate within this [Installation](#installation) section for different operating systems or particularly complicated installations. Keep in mind, though, that the more complicated the installation process is, the more likely that users will encounter difficulties and give up.


## Quick start

Nobody wants to read long explanations about how to use your software before they can try it, especially while they are still deciding _whether_ to try it. A [Quick start](#quick-start) section right after the installation instructions can help readers figure out what's involved.

Explain the minimal configuration (if any) required to use the software, then provide the simplest example or command that demonstrates actual functionality implemented by your software. If your software is command-line oriented, provide examples (again in [code blocks](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks) in the Markdown file).

```sh
yoursoftware argument1 argument2
```

If your software is not command-line oriented, providing static screenshots annotated with arrows or other guidance is a good approach to showing readers how to use the software. The temptation to use embedded videos or animated GIFs here may be strong, but it is better to resist it, or at least to use them only as a way to _augment_ and not replace written explanations. It takes people longer to watch a video than to skim textual descriptions on a page, which can annoy potential users; moreover, some users turn off video players in their browsers and may not even see the embedded video content.


## Usage

The [Usage](#usage) section would explain in more detail how to run the software, what kind of output or behavior to expect, and so on. It would cover basic operations as well as more advanced uses.

Some of the information in this section will repeat what is in the [Quick start](#quick-start) section. This repetition is unavoidable, but also, not entirely undesirable: the more detailed explanations in this [Usage](#usage) section can help provide more context as well as clarify possible ambiguities that may exist in the more concise [Quick start](#quick-start) section.

If your software is complex and has many features, it may be better to create a dedicated website for your documentation (e.g., in [GitHub Pages](https://pages.github.com), [Read the Docs](https://about.readthedocs.com), or similar) rather than to cram everything into a single linear README file. In that case, the [Usage](#usage) section can be shortened to just a sentence or two pointing people to your documentation site.


### Basic operation

When learning how to use anything but the simplest software, new users may appreciate beginning with basic features and modes of operation. If your software has a help system of some kind (e.g., in the form of a command-line flag such as `--help`, or a menu item in a GUI), explaining it is an excellent starting point for this section.

The basic approach for using this README file is as follows:

1. Copy the [README source file](https://raw.githubusercontent.com/mhucka/readmine/main/README.md) to your repository
2. Delete the body text but keep the section headings
3. Replace the title heading (the first line of the file) with the name of your software
4. Save the resulting skeleton file in your version control system
5. Continue by writing your real README content in the file

The first paragraph in the README file (under the title at the top) should summarize your software in a concise fashion, preferably using no more than one or two sentences as illustrated by the circled text in the figure below.

<p align="center">
<img alt="Screenshot showing the top portion of this file on the web." width="80%" src="https://raw.githubusercontent.com/mhucka/readmine/main/.graphics/screenshot-top-paragraph.png"><br>
<em>Figure: Screenshot showing elements of the top portion of this file.</em>
</p>

The space under the first paragraph and _before_ the [Table of Contents](#table-of-contents) is a good location for optional [badges](https://github.com/badges/shields), which are small visual tokens commonly used on GitHub repositories to communicate project status, dependencies, versions, DOIs, and other information. (Two example badges are shown in the figure above, under the circled text.) The particular badges and colors you use depend on your project and personal tastes.


### More options

Some projects need to communicate additional information to users and can benefit from additional sections in the README file. It's difficult to give specific instructions here – a lot depends on your software, your intended audience, etc. Use your judgment and ask for feedback from users or colleagues to help figure out what else is worth explaining.


## Known issues and limitations

In this section, summarize any notable issues and/or limitations of your software. If none are known yet, this section can be omitted (and don't forget to remove the corresponding entry in the [Table of Contents](#table-of-contents) too); alternatively, you can leave this section in place and write something along the lines of "none are known at this time".


## Getting help

Inform readers how they can contact you, or at least how they can report problems they may encounter. This could take the form of a request to use the issue tracker on your repository. Some projects have associated discussion forums or mailing lists, and this section is a good place to mention those.


## Contributing

If your project accepts open-source contributions, this is where you can welcome contributions and explain to readers how they can go about it. Mention the [`CONTRIBUTING.md`](CONTRIBUTING.md) file in your repository, if you have one.


## License

This section should state any copyright asserted on the project materials as well as the terms of use for the software, files and other materials found in the project repository.

_This_ README file is itself distributed under the terms of the [Creative Commons 1.0 Universal license (CC0)](https://creativecommons.org/publicdomain/zero/1.0/). The license applies to this file and other files in the [GitHub repository](http://github.com/mhucka/readmine) hosting this file. This does _not_ mean that you, as a user of this README file in your software project, must also use CC0 license!  You may use whatever license for your work you prefer, or whatever you are required to use by your employer or sponsor.


## Acknowledgments

This final section is where you should acknowledge funding and/or institutional support, prior work that influenced or inspired your project, resources that you used (such as other people's software), important contributions from other people, and anything else that deserves mention. After all, nothing is truly done in isolation; everything is built on top of something, and we all owe debts to other projects and people who helped us, supported us, and influenced us.

For example, in the process of developing this file, I used not only my own ideas: I read many (sometimes contradictory) recommendations for README files, examined real READMEs in actual use, and tried to distill the best ideas into the result you see here. Sources included the following:

* [Readme Driven Development](http://tom.preston-werner.com/2010/08/23/readme-driven-development.html)
* [How to Write a Good README](https://dev.to/merlos/how-to-write-a-good-readme-bog)
* [How To Write A Great README](https://thoughtbot.com/blog/how-to-write-a-great-readme)
* [How to Write an Awesome Readme](https://dev.to/documatic/how-to-write-an-awesome-readme-cfl)
* [Readme Best Practices](https://github.com/jehna/readme-best-practices)
* [Art of README](https://github.com/noffle/art-of-readme)
* [Making a useful README file for research projects](http://jonathanpeelle.net/making-a-readme-file)
* [Tips for Making your GitHub Profile Page Accessible](https://github.com/orgs/community/discussions/64778)
* [Make a README](https://www.makeareadme.com)
* [ReadMe.so](https://readme.so)
* [common readme](https://github.com/noffle/common-readme)
* [Standard Readme](https://github.com/RichardLitt/standard-readme)
* [CFPB Open Source Project Template Instructions](https://github.com/cfpb/open-source-project-template)
* [README-Template.md](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [open-source-template](https://github.com/davidbgk/open-source-template/)
* [Feedmereadmes: A README Help Exchange](https://github.com/lappleapple/feedmereadmes)
* [Awesome README List](https://github.com/matiassingers/awesome-readme)
* [Top ten reasons why I won't use your open source project](https://changelog.com/posts/top-ten-reasons-why-i-wont-use-your-open-source-project)
