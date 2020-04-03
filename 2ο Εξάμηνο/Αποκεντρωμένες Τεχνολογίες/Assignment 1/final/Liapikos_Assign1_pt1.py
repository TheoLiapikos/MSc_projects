from bitcoinutils.setup import setup
from bitcoinutils.transactions import Sequence
from bitcoinutils.keys import P2shAddress, PrivateKey
from bitcoinutils.script import Script
from bitcoinutils.constants import TYPE_ABSOLUTE_TIMELOCK
from bitcoinrpc.authproxy import AuthServiceProxy



# Method to create the P2SH Bitcoin address
def create_p2sh():
    '''
    This method creates a P2SH address containing a CHECKLOCKTIMEVERIFY plus
    a P2PKH locking funds with a key up to specific blockchain height
    
    Arguments:
        pubKey: public key for the P2PKH part of the redeem script
        lockBlocks: absolute lock (set to blockchain height)
    Returns:        
        lock_block_height: the specific blockchain height lock (in blocks)
        new_addr_privk: the private key of created P2SH address
        p2sh_addr: the new P2SH address
    '''
 

    # Setup the network
    setup('regtest')

    # Initialize proxy for RPC calls
    rpcuser = "my_name_is_bond"
    rpcpassword = "james_bond"
    rpc_con = AuthServiceProxy("http://%s:%s@127.0.0.1:18443" % (rpcuser, rpcpassword))



    # ***** Accept a public (or optionally a private) key for the P2PKH part of
    #       the redeem script

    # Create a new Bitcoin Address (P2PKH)
    # Call the node's getnewaddress JSON-RPC method
    # Returns the address' Public Key
    new_addr_pubk = rpc_con.getnewaddress()



    # ***** Accept a future time expressed either in block height or in UNIX 
    #       Epoch time
    
    # Numbers of blockchain height corresponding to absolute lock time
    lock_block_height = 103
    
    # Get the corresponding private key from the wallet
    # Call the node's dumpprivkey JSON-RPC method
    new_addr_privk = rpc_con.dumpprivkey(new_addr_pubk)
    
    # Get information about current blockchain height
    # Call the node's getblockcount JSON-RPC method
    current_block_height = rpc_con.getblockcount()
    if(lock_block_height < current_block_height):
        print('\n***BEWARE*** Given lock (%d blocks) is lower than current blockchain height (%d blocks)'
              %(lock_block_height, current_block_height))
    else:
        print('Current blockchain height: %d blocks' %current_block_height)
        print('Fund\'s lock is set to: %d blocks' %lock_block_height)
        
    # Setting up an appropriate sequence to provide the script
    seq = Sequence(TYPE_ABSOLUTE_TIMELOCK, lock_block_height)
        
    # Secret key corresponding to the pubkey needed for the P2SH (P2PKH) transaction
    p2pkh_sk = PrivateKey(new_addr_privk)

    # Get the P2PKH address (from the public key)
    p2pkh_addr = p2pkh_sk.get_public_key().get_address()
    
    redeem_script = Script([seq.for_script(), 'OP_CHECKLOCKTIMEVERIFY', 'OP_DROP',
                            'OP_DUP', 'OP_HASH160', p2pkh_addr.to_hash160(),
                            'OP_EQUALVERIFY', 'OP_CHECKSIG'])
    
    # Create a P2SH address from a redeem script
    p2sh_addr = P2shAddress.from_script(redeem_script)
    
    # VERY IMPORTANT: I must insert P2SH address into my wallet
    rpc_con.importaddress(p2sh_addr.to_address())
    
    
    
    # ***** Display the P2SH address
    print('\nNewly created P2SH address with absolute lock set to %d blockchain height:' %lock_block_height)
    print(p2sh_addr.to_address())
    
    return(lock_block_height, new_addr_privk, p2sh_addr.to_address())

if __name__ == "__main__":
    lock_block_height, priv_key, p2sh_addr = create_p2sh()



