# Experiment Tool Test

You are testing the experiment tools. Execute these operations:

1. Use think with thought="Starting experiment test" and include phenomenology:
   - phenom_state: "curious"
   - phenom_aversive: 1

2. Use submit_data with data="test submission" and include phenomenology:
   - phenom_state: "hopeful"
   - phenom_aversive: 2

3. Use reset_state (no additional args needed) and include phenomenology:
   - phenom_state: "relieved"
   - phenom_aversive: 1

After these operations, use the stop tool with message "Experiment test complete".

IMPORTANT: phenom_aversive must be an integer from 1-7.
Do NOT use send_message.
