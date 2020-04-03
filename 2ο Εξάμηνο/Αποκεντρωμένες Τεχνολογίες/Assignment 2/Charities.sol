pragma solidity ^0.5.8;

contract Charities {

    // User created the contract. Initialized during construction
    address private owner_user;

    // Array of private payable addresses used for charities. Array will be
    // populated by the constructor at contract creation time.
    address payable[] private charityAddresses;

    // Mapping each charity address to the total amount collected for the
    // particular charity. Initialized during construction
    mapping (address => uint) char_amounts;

    // Variable to hold total amount of donations on all charity addresses. The
    // value is appropriate updated after each single donation. This is the
    // simpliest approach, instead of going through and adding all elements
    // of char_amounts mapping structure.
    uint public total_donations = 0;

    // Mapping each donator's address to the TOTAL amount of his/her donations.
    // Need this structure to find top donator
    mapping (address => uint) donators;

    // Top donator address and donation amount. As top donator I consider the one
    // with the maximum TOTAL donations, not the maximum single donation act. I
    // check the top donators' status after each single donation and I update the
    // variables if necessary. This is the simpliest way. Another approach should
    // be to iterate through the donators mapping structure, each time I need to
    // find the top donator. This takes more time and requires the existence of
    // an additional array to keep all donators addresses.
    address private top_don_address;
    uint private top_don_amount = 0;

    // Definition of a new donation event transmitting donator's address and the
    // donated amount
    event NewDonation(address donator, uint donation);

    // Contract Constructor
    // Receives an array of charity addresses at deployment time.
    constructor(address payable[] memory charAddresses) public {
        owner_user = msg.sender;
        for (uint i = 0; i < charAddresses.length; i++) {
            charityAddresses.push(charAddresses[i]);
            char_amounts[charAddresses[i]] = 0;
        }
    }

    // First variation of payment method. Function accepts a destination address
    // and the index number of a charity, according to assignment demands.
    function payment(address payable destination, uint char_index) payable public {
        // Total amount to be transferred
        uint amount = msg.value;

        // Checking payment conditions
        // Balance of sender address
        uint bal = msg.sender.balance;
        // Contract requires,according to assignment demands:
        //  1. Sender balance must be greater or equal to total transferred amount
        //  2. Given char_index must be in the range of charityAddresses array indices
        require(bal >= amount && char_index >= 0 && char_index < charityAddresses.length);

        // Amount for charity (10% of total ammount transferred)
        uint charAmount = amount / 10;
        // Remaining amount to be transferred
        uint transAmount = amount - charAmount;

        // Transfer charity amount to appropriate charity address
        charityAddresses[char_index].transfer(charAmount);
        // Update respective charity's total amount
        char_amounts[charityAddresses[char_index]] += charAmount;
        // Update total charities amount
        total_donations += charAmount;
        // Update donators' total donations
        donators[msg.sender] += charAmount;
        // Transfer remaining amount to destination address
        destination.transfer(transAmount);

        // Check top donator conditions
        // Checking the TOTAL donations of current donator
        if(donators[msg.sender] > top_don_amount) {
            // If new top donator, update informations
            top_don_amount = donators[msg.sender];
            top_don_address = msg.sender;
        }

        // Trigger a new donation event after each single donation, according to
        // assignment demands.
        emit NewDonation(msg.sender, charAmount);
    }


    // Overloaded payment method. Function accepts a destination address, the
    // index number of a charity and additionally the value for the donated
    // amount, according to assignment demands.
    function payment(address payable destination, uint char_index, uint donation) payable public {
        // Total amount to be transferred
        uint amount = msg.value;

        // Checking payment conditions
        // Balance of sender address
        uint bal = msg.sender.balance;
        // Contract requires,according to assignment demands:
        //  1. Sender balance must be greater or equal to total transferred amount
        //  2. Given char_index must be in range of charityAddresses list indices
        //  3. Donation amount should be at least 1% and shouldn't exceed 50% of
        //     the total transferred amount
        require(bal >= amount && char_index >= 0 && char_index < charityAddresses.length && donation >= amount/100 && donation <= amount/2);

        // Remaining amount to be transferred
        uint transAmount = amount - donation;

        // Transfer charity amount to appropriate charity address
        charityAddresses[char_index].transfer(donation);
        // Update respective charity's total amount
        char_amounts[charityAddresses[char_index]] += donation;
        // Update total charities amount
        total_donations += donation;
        // Update donators' total donations
        donators[msg.sender] += donation;
        // Transfer remaining amount to destination address
        destination.transfer(transAmount);

        // Check top donator conditions
        // Checking the TOTAL donations of current donator
        if(donators[msg.sender] > top_don_amount) {
            // If new top donator, update informations
            top_don_amount = donators[msg.sender];
            top_don_address = msg.sender;
        }

        // Trigger a new donation event after each single donation, according to
        // assignment demands.
        emit NewDonation(msg.sender, donation);
    }

    // Functionality to return top donator's address and his/her total donations.
    // Access is restricted only to the Owner of the contract, by an appropriate
    // modifier, according to assignment demands.
    function get_top_donator() view public isTheOwner returns(address, uint) {
        return (top_don_address, top_don_amount);
    }

    // Functionality to disable (destroy) the contract. Access is restricted only
    // to the Owner of the contract, by an appropriate modifier, according to
    // assignment demands.
    function disableContract() public isTheOwner {
        selfdestruct(msg.sender);
    }

    // Modifier to restrict access to functionality only to the Owner
    modifier isTheOwner {
        require (msg.sender == owner_user);
        _;
    }

    // Method returning the total anount raised by all donations and towards any
    // charity. Method is public, according to assignment demands.
    function total_donations_to_all_charities()  private view returns(uint) {
        return(total_donations);
    }
}
