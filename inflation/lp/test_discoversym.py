
from discoversym import *
import unittest

def get_bilocal_network():    
    g = NetworkGraph(n_sources=2,n_agents=3)
    g[0,0] = True
    g[0,1] = True

    g[1,1] = True
    g[1,2] = True

    return g

def get_triangle_network():
    g = NetworkGraph(n_sources=3,n_agents=3)
    g[0,0] = True
    g[0,1] = True

    g[1,1] = True
    g[1,2] = True

    g[2,2] = True
    g[2,0] = True

    return g

def apply_agent_perm(p: Distr, agent_perm: AgentPerm) -> Distr:
    """This function is a test function to test that we are really doing first agent_perm and then io_perm"""

    new_p = Distr(p.get_shape())
    n_agents = p.get_n_agents()

    for indices in np.ndindex(*p.get_shape()):
        perm_indices = [0] * (2*n_agents)

        for i_agent in range(n_agents):
            perm_indices[         agent_perm[i_agent]] = indices[         i_agent]
            perm_indices[n_agents+agent_perm[i_agent]] = indices[n_agents+i_agent]
        
        new_p[tuple(perm_indices)] = p[indices]
    
    return new_p

def apply_io_perm(p: Distr, io_perm: IOPerm) -> Distr:
    """This function is a test function to test that we are really doing first agent_perm and then io_perm"""
    
    new_p = Distr(p.get_shape())
    n_agents = p.get_n_agents()

    for indices in np.ndindex(*p.get_shape()):
        perm_indices = [0] * (2*n_agents)

        for i, the_perm in enumerate(io_perm):
            perm_indices[i] = the_perm[indices[i]]
        
        new_p[tuple(perm_indices)] = p[indices]

    return new_p

class TestDiscoverSymmetries(unittest.TestCase):

    def setUp(self):
        pass

    def test_graph_basic(self):
        g = NetworkGraph(n_sources=4,n_agents=5)
        self.assertEqual(g.get_n_sources(),4)
        self.assertEqual(g.get_n_agents(),5)

        g_bis = NetworkGraph(shape=[4,5])
        self.assertEqual(g.get_n_sources(),4)
        self.assertEqual(g.get_n_agents(),5)

        g_ter = NetworkGraph(shape=g_bis.get_shape())

        self.assertEqual(g_bis,g_ter)
        
        
    def test_automorphisms(self):
        for g,expected_n in [(get_bilocal_network(),2), (get_triangle_network(),6)]:
            automorphisms = g.get_automorphisms()

            if False:
                print(g)
                print("---- Looking for graph automorphisms... ----")
                for agent_perm,source_perm in automorphisms:
                        print("* found an automorphism: agents go",agent_perm,"and sources go",source_perm)
                print("--------------------------------------------")
            else:
                self.assertEqual(len(automorphisms),expected_n,"unexpected number of automorphisms")


    def test_distr(self):
        g = get_triangle_network()
        p = Distr([2,2,2,1,1,1])
        p[0,0,1,0,0,0] = 1./2.
        p[1,1,0,0,0,0] = 1./2.

        # print(g)
        # print(p)

        symmetries = p.get_symmetries(g)

        # Expect something isomorphic to S_2 \times S_3 (bit exchange + party exchange) 
        self.assertEqual(len(symmetries),12)
        
        # print("---- Finding distrib symmetries... ----")
        for sym in symmetries:
            # print("Found one:",sym)
            # Expect that symmetries are applied as first agent_perm, then io_perm
            self.assertTrue(
                np.isclose(p.mat, apply_io_perm(apply_agent_perm(p, sym.agent_io_perm.get_agent_perm()), sym.agent_io_perm.get_io_perm()).mat ).all()
            )
        # print("---------------------------------------")


if __name__=="__main__":
    unittest.main()

