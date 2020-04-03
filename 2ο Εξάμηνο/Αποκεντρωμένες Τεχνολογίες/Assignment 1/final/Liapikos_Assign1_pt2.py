from bitcoinutils.setup import setup
from bitcoinutils.transactions import Transaction, TxInput, TxOutput, Locktime
from bitcoinutils.keys import PrivateKey
from bitcoinutils.script import Script
from bitcoinutils.transactions import Sequence
from bitcoinutils.constants import TYPE_ABSOLUTE_TIMELOCK
from bitcoinrpc.authproxy import AuthServiceProxy
import random
import sys
from Liapikos_Assign1_pt1 import create_p2sh


# Setup the network
setup('regtest')

# Get a node proxy using default host and port
#proxy = NodeProxy('my_name_is_bond', 'james_bond').get_proxy()
# Initialize proxy for RPC calls
rpcuser = "my_name_is_bond"
rpcpassword = "james_bond"
rpc_con = AuthServiceProxy("http://%s:%s@127.0.0.1:18443" % (rpcuser, rpcpassword))



# ***** Accept a future time, expressed either in block height or in UNIX Epoch time
# ***** Accept a private key (to recreate the redeem script as above and also
#       use to unlock the P2PKH part)
# ***** Accept a P2SH address to get the funds from (the one created by the first script)

# I import the appropriate method, shown in part1 of the Assignement, to create
# a new P2SH here. All requested values are created and returned by the method.

# Create a new P2SH address using method from part1
lock_block_height, priv_key, p2sh_addr = create_p2sh()



# ***** Τest your scripts by sending some funds to the P2SH address you created

def send_bts_to_address(address):
    '''
    This script sends various amounts of bitcoins to a specific address, using
    up to 5 (randomly selected) different transactions.
    
    Arguments:
        address: Bitcoin address to receive the payments
    Returns:
        Prints the ampunt (in BTCs) and the ID of each payment transaction.
        Prints the total amount transferred and the total number of transactions
    '''
    
    print('\nPayments to P2SH address:')
    #Random number of payment transactions (up to 5)
    num_txs = random.randint(2,5)
    # Total amount sent to address
    amount = 0
    for i in range(num_txs):
        # Random amount of bts to be sent (from 5 to 10 BTCs)
        bts = round(10*random.random(),8)
        print('Payment %d: %.8f BTCs' %(i+1, bts))
        amount += bts
        print(rpc_con.sendtoaddress(address, bts))
    
    print('Total amount sent to P2SH address: %.8f BTCs (%d TXs)' %(amount, num_txs))
    # Mine 1 block in order the transactions to take effect
    rpc_con.generate(1)


# Lets send some BTs to P2SH address
# I assume that blockchain is initially empty, so i have first to create
# at least 101 blocks to receive the first mining fees, in order to be able
# to make the payment transactions.
print('\nMining first 101 blocks')
rpc_con.generate(101)

# Create random # of payments to P2SH address 
send_bts_to_address(p2sh_addr)

# Check mempool for pending transactions and P2SH address for total received amount
print('\nPending transactions in mempool: %d' %len(rpc_con.getrawmempool()))
print('\nTotal amount received by P2SH address: %.8f BTCs' %rpc_con.getreceivedbyaddress(p2sh_addr))



# ***** Check if the P2SH address has any UTXOs to get funds from

def find_UTXOs_of_address(address):
    '''
    This method scans all UTXOs of all addresses in the wallet and returns only 
    the ones having a specific address as output.
    
    Arguments:
        address: the specific address to search for UTXOs
    Returns:
        a dictionary containing the IDs (keys) and the funds (values) of all
        UTXOS found for the specific address
    '''
    # Total UTXOs from wallet
    total_utxos = rpc_con.listunspent()
    addr_utxos = {}
    amount = 0
    # for each one of the transactions above
    for utx in total_utxos:
        # Get the output address of the transaction
        out_addr = utx['address']
        # If equals to P2SH address
        if(out_addr == p2sh_addr):
            amount += utx['amount']
            addr_utxos[utx['txid']] = utx['amount']

    print('\nUTXOs for address %s: ' %address)
    print('Found %d UTXOs corresponding to %.8f BTCs' %(len(addr_utxos), amount))
    return(addr_utxos)


# Lets find the UTXOs corresponding to P2SH address
p2sh_utxos = find_UTXOs_of_address(p2sh_addr)



# ***** Accept a P2PKH address to send the funds to

# Instead of using an existing P2PKH address I created a new one (for training reasons)

# Returns the address' Public Key
rec_addr_pubk = rpc_con.getnewaddress()
# Address' Private Key
rec_addr_privk = rpc_con.dumpprivkey(rec_addr_pubk)
# Secret key corresponding to the pubkey
rec_p2pkh_sk = PrivateKey(rec_addr_privk)
# Get the P2PKH address (from the public key)
rec_p2pkh_addr = rec_p2pkh_sk.get_public_key().get_address()



# ***** Calculate the appropriate fees with respect to the size of the transaction

# Didn't manage to connect to an on-line service to automatic calculate fees
# based on transaction's size. So I used an empirical formula to calculate
# transaction's size (in bytes) and then to calculate the total fees using an
# average fees per byte.
def calc_tx_fee(inputs, outputs, btcs_per_byte=50e-8):
    '''
    Method to calculate the total fees for a transaction. First calculates the
    transaction's size using an empirical formula (found on-line (1)) based on
    number of transaction's inputs and outputs. The formula is as follows:
        (inputs*180) + (outputs*34) + 10 bytes
    Then multiply the size by an average fees per byte, found on-line (50 satochis/byte)
    
    Arguments:
        inputs: number of transaction's inputs
        putputs: number of transaction's outputs
        btcs_per_byte: average fees per transaction's byte (in BTCs)
    Returns:
        total calculated fees for transaction (in BTCs)
        
    (1) https://news.bitcoin.com/how-to-calculate-bitcoin-transaction-fees-when-youre-in-a-hurry/
    '''
    # Calculate transaction's size
    size = (inputs*180) + (outputs*34) + 10
    # Calculate total fees (bytes * 50 sats/byte)
    total_fee = size*btcs_per_byte # in BTCs
    return total_fee


# I will create a transaction with a single input and a single output
fixed_fee = calc_tx_fee(1, 1)
print('\nCalculated fees for transaction: %.8f BTCs' % fixed_fee)



# ***** Send all funds that the P2SH address received to the P2PKH address provided

# Didn't manage to sign a multi-input transaction so instead I will create a
# single-input transaction spending only the first UTXO found for the P2SH address

# Prepare the elements needed for the transaction

# Create the Inputs
# I will use the data from the first UTXO
in_txid, in_amount = list(p2sh_utxos.items())[0]
vout = 0
# Set sequence values for inputs as in part 1
seq = Sequence(TYPE_ABSOLUTE_TIMELOCK, lock_block_height)
txin = TxInput(in_txid, vout, sequence=seq.for_input_sequence())

# Create the Output
# I will spend the whole input amount minus a fixed fee. No change output needed
# Clear amount to transfer
in_amount = float(in_amount)
out_amount = round(in_amount-fixed_fee, 8)
txout = TxOutput(out_amount, rec_p2pkh_addr.to_script_pub_key())

# Lock time (in blocks) to be used in transaction
lock_time = Locktime(lock_block_height).for_transaction()

# Compose transaction (Raw Unsigned Transaction)
p2sh_out_tx = Transaction(inputs=[txin], outputs=[txout], locktime=lock_time)



# ***** Display the Raw Unsigned Transaction

# Print Raw Unsigned Transaction
r_u_t = p2sh_out_tx.serialize()
print('\nRaw Unsigned Transaction: %s' %r_u_t)



# *** Sign the transaction

# Create signature to unlock inputs
# I have to rebuild redeem_script from P2SH private key
# Secret key of P2SH address
p2pkh_sk = PrivateKey(priv_key)
# Get the P2PKH public key
p2pkh_pk = p2pkh_sk.get_public_key().to_hex()
# Get the P2PKH address (from the public key)
p2pkh_addr = p2pkh_sk.get_public_key().get_address()
# Create the redeem script
redeem_script = Script([seq.for_script(), 'OP_CHECKLOCKTIMEVERIFY', 'OP_DROP',
                        'OP_DUP', 'OP_HASH160', p2pkh_addr.to_hash160(),
                        'OP_EQUALVERIFY', 'OP_CHECKSIG'])

# Signature
sign = p2pkh_sk.sign_input(p2sh_out_tx, 0, redeem_script)

# Unlock transaction's input using signature, P2SH public key and redeem_script
txin.script_sig = Script([sign, p2pkh_pk, redeem_script.to_hex()])



# ***** Display the raw signed transaction

# Print Raw Signed Transaction
r_s_t = p2sh_out_tx.serialize()
print('\nRaw Signed Transaction: %s' %r_s_t)



# *** Display the transaction ID

# Signed Transaction ID
r_s_t_id = p2sh_out_tx.get_txid()
print('\nRaw Signed Transaction ID: %s' %r_s_t_id)



# ***** Verify that the transaction is valid and will be accepted by the Bitcoin nodes
    
def verify_tx(raw_signed_tx):
    '''
    This method verifies if a signed raw transaction would be accepted by mempool
    by checking if the transaction violates the consensus or policy rules.
    
    Arguments:
        raw_signed_tx: the specific transaction (serialized, hex-encoded)
    Returns:
        system report
    '''
    return(rpc_con.testmempoolaccept([raw_signed_tx]))
    

# BEWARE! Transaction's inputs are locked for ABSOLUTE blockchain height up to
# lock_block_height blocks. Trying to send the transaction in a lower blockchain
# will cause the proxy to collapse.
current_block_height = rpc_con.getblockcount()
if(current_block_height<lock_block_height):
    # Mine blocks to reach desired blockchain height
    rpc_con.generate(lock_block_height - current_block_height)


ver = verify_tx(r_s_t)
print('\nTransaction verification:')
print(ver)
if(ver[0]['allowed'] == False):
    print('Transaction rejected (reject-reason: %s). Transaction broadcast cancelled.\n' %ver[0]['reject-reason'])
    sys.exit(0)



# ***** If the transaction is valid, send it to the blockchain

def broadcast_tx(raw_signed_tx):
    '''
    This method verifies if a signed raw transaction would be accepted by mempool
    by checking if the transaction violates the consensus or policy rules.
    
    Arguments:
        raw_signed_tx: the specific transaction (serialized, hex-encoded)
    '''
    # When all condition met, send the transaction in blockchain
    print('\nTransaction sent successfully in blockchain:',rpc_con.sendrawtransaction(raw_signed_tx))
    print('%.8f BTCs sent to address: %s' %(out_amount, rec_p2pkh_addr.to_address()))



if(ver[0]['allowed'] == True):
    print('Transaction seems ok!. Proceed to transaction broadcast.')
    broadcast_tx(r_s_t)



# ***** Extra verification that the BTCs were received by recipient P2PKH address
# Check mempool for pending transactions before mining
print('\nPending transactions in mempool before mining: %d' %len(rpc_con.getrawmempool()))
# Mine one block to validate the transaction
rpc_con.generate(1)
# Check mempool for pending transactions after mining
print('Pending transactions in mempool after mining: %d' %len(rpc_con.getrawmempool()))

# Check recipient P2PKH address for received BTCs
print('\nChecking recipient P2PKH address for received BTCs:')
print('Recipient address received %.8f BTCs.' %rpc_con.getreceivedbyaddress(rec_p2pkh_addr.to_address()))



