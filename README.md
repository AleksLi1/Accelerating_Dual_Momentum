# Accelerating Dual Momentum
This is an adaptation of the Accelerating Dual Momentum tactical asset allocation model which was originally described [here](https://allocatesmartly.com/taa-strategy-accelerating-dual-momentum/).

Some preliminary performance stats:

![strat](https://i.ibb.co/yXQPt8s/2.png)

The rules are as follows:

- This strategy allocates 100% of the portfolio to a single asset each month.
- At the close on the first trading day of the month, calculate a “momentum score” for two asset classes: the S&P 500 (represented by SPY) and small cap equities (VBR), by averaging each asset’s 1, 3 and 6-month total return.
- If the momentum score of SPY > VBR and > 0, go long SPY at the close.
- If the momentum score of VBR > SPY and > 0, go long VBR at the close.
- If neither condition is true, go long either long-term US Treasuries (TLT) or US TIPS (TIP), whichever has the highest 1-month return.
- Hold position until the first trading day of the following month.

![stats](https://i.ibb.co/D1ZrZQ6/1.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact
If you would like to get in touch, my email is aleksandras.v.liauska@bath.edu.