# Slate Simulator

## Actions
- x, y, z values
  - x, 0.0 - 4.0
  - y, 0.0 - 3.0
  - z, 0.0 - 2.0

## Context
- OS: Mac, Windows
- Connection: wifi, wired
- Region: US, CA

## Reward Function

Given x, y and z, the following coefficients can be used to calculate the reward value. The reward value must then be rescaled to the desired range.

- Coefficients are displayed in x, y, z, xy, xz, yz form.

| Context | c1 | c2 | c3 |c4 |c5 |c6 |
| --- | --- | --- | --- | --- | --- | --- |
| Mac, wifi, US | 0.37755595 | 0.88794085 | 0.78759054 | 0.60708194 | 0.92570716 | 0.35602915 |
| Mac, wifi, CA | 0.19113076 | 0.17363948 | 0.64172931 | 0.5095073  | 0.45841506 | 0.43078203 |
| Mac, wired, US | 0.78266802 | 0.2267633 |  0.95940249 | 0.33171948 | 0.36201023 | 0.36354627 |
| Mac, wired, CA | 0.55245693 | 0.95071475 | 0.21295371 | 0.35589226 | 0.25239824 | 0.6135975] |
| Windows, wifi, US | 0.14937779 | 0.46339031 | 0.49216011 | 0.14819576 | 0.47126218 | 0.26317773 |
| Windows, wifi, CA | 0.60262362 | 0.4632763 |  0.21433892 | 0.65638386 | 0.19664801 | 0.30497455 |
| Windows, wired, US | 0.67744709 | 0.88068832 | 0.7702806 |  0.7884114  | 0.87577138 | 0.41156948 |
| Windows, wired, CA | 0.87995436 | 0.10853902 | 0.24386487 | 0.14241173 | 0.30777027 | 0.14954826 |

```
reward = c1*x + c2*y + c3*z + c4*x*y + c5*x*z + c6*y*z
