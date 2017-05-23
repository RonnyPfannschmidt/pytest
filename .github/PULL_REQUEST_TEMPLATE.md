Thanks for submitting a PR, your contribution is really appreciated!

Here's a quick checklist that should be present in PRs:

- [ ] Add a new news fragment into the news folder
  * name it $issue_id.$type
  * if you dont have an issue_id change it to the pr id after creating the pr
  * type is one of removal, feature, bugfix, vendor, doc, trivial
- [ ] Target: for bbugfix, vendor, doc or trivial fixes, target `master`; for removals or features target `features`;
- [ ] Make sure to include reasonable tests for your change if necessary

Unless your change is trivial documentation fix (e.g.,  a typo or reword of a small section) please:

- [ ] Add yourself to `AUTHORS`;
