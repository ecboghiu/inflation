"""
This file contains classes related to discovering the symmetries of a distribution with respect to a graph.

TODOS:
- Get nice output repr
- Check for matching of cardinalities!!!
- Check for matching inflation levels!!!
- Parametrize the isclose test nicely and visibly
"""

import math
import numpy as np
import itertools
from typing import List, Tuple, Iterator

# ---------------
# -- Utilities --
# ---------------

INT_TO_NAME = ["A","B","C","D","E"]
INT_TO_SETTING = ["X","Y","Z","U","V"]
INT_TO_SOURCE = ["alpha"," beta","gamma","delta","epsilon"]

# ------------------
# -- Permutations --
# ------------------

class AgentPerm:
    """A simple container for agent permutations"""

    def __init__(self,perm: List[int]):
        self.perm = perm

    def get_n_agents(self) -> int:
        return len(self.perm)

    def agent_is_trivial(self, i_agent: int) -> bool:
        """Return True if agent is unchanged"""
        return self[i_agent] == i_agent

    def __getitem__(self,i: int) -> int:
        return self.perm[i]

    def __setitem__(self,i: int,val: int):
        self.perm[i] = val

    def __str__(self) -> str:
        ret = ""
        for a, pi_a in enumerate(self.perm):
            ret += INT_TO_NAME[a] + "->" + INT_TO_NAME[pi_a]
            if a < len(self.perm)-1:
                ret += ", "
        return ret

    def __iter__(self):
        return (pi_i for pi_i in self.perm)

class SourcePerm:
    """A simple container for source permutations"""

    def __init__(self,perm: List[int]):
        self.perm = perm
    
    def get_array(self) -> List[int]:
        return self.perm

    def source_is_trivial(self, i_source: int) -> bool:
        """Return True is source is unchanged"""
        return self[i_source] == i_source
    
    def __getitem__(self,i: int) -> int:
        return self.perm[i]
    
    def __setitem__(self,i: int,val: int):
        self.perm[i] = val
    
    def __str__(self) -> str:
        ret = ""
        for s, pi_s in enumerate(self.perm):
            ret += INT_TO_SOURCE[s] + "->" + INT_TO_SOURCE[pi_s]
            if s < len(self.perm)-1:
                ret += ", "
        return ret

class IOPerm:
    """A class that stores a permutation of all inputs and outputs, i.e., a list of permutations"""

    def __init__(self,perms: List[List[int]]):
        self.perms = perms

    def get_n_agents(self) -> int:
        return int(len(self.perms)/2)

    def get_operm(self, i_agent: int) -> List[int]:
        """This method returns the i_agent'th output permutation, i.e., self.perms[i_agent]"""
        return self.perms[i_agent]
    
    def get_iperm(self, i_agent: int) -> List[int]:
        """This method returns the i_agent'th input permutation, i.e., self.perms[n_agents+i_agent]"""
        return self.perms[self.get_n_agents() + i_agent]
    
    def get_ocard_iter(self, i_agent: int) -> Iterator[int]:
        return iter( range(len(self.get_operm(i_agent))) )
    
    def get_icard_iter(self, i_agent: int) -> Iterator[int]:
        return iter( range(len(self.get_iperm(i_agent))) )
        
    def agent_is_trivial(self, i_agent: int) -> bool:
        """Return True if both input & output permutation of an agent is trivial"""
        for i, pi_i in enumerate(self.get_operm(i_agent)):
            if i != pi_i:
                return False
        for i, pi_i in enumerate(self.get_iperm(i_agent)):
            if i != pi_i:
                return False
        return True

    def event_is_trivial(self, i_agent: int, setting: int, outcome: int) -> bool:
        """Return True if both setting & outcome are mapped to the same thing"""
        return (self.get_iperm(i_agent)[setting] == setting) and (self.get_operm(i_agent)[outcome] == outcome)

    def __setitem__(self,index: int,perm: List[int]):
        self.perms[index] = perm
    
    def __getitem__(self,index: int) -> List[int]:
        return self.perms[index]

    def __len__(self) -> int:
        return len(self.perms)

    def __str__(self) -> str:
        ret = ""

        for i in range(2*self.get_n_agents()):
            if i < self.get_n_agents():
                ret +=    INT_TO_NAME[i]
            else:
                ret += INT_TO_SETTING[i-self.get_n_agents()]
            
            ret += ": ("

            for j, pi_j in enumerate(self.perms[i]):
                ret += str(j) + "->" + str(pi_j)
                if j < len(self.perms[i])-1:
                    ret += ", "
            ret += ")"

            if i < 2*self.get_n_agents()-1:
                ret += ", "
        
        return ret

    def __iter__(self):
        return (the_perm for the_perm in self.perms)

class AgentIOPerm:
    """A class that stores a permutation of FIRST agent_perm and THEN io_perm"""

    def __init__(self,agent_perm: AgentPerm,io_perm: IOPerm):
        self.agent_perm = agent_perm
        self.io_perm = io_perm

    def __str__(self) -> str:
        return "agents " + str(self.agent_perm) + " then io " + str(self.io_perm)

    def __repr__(self) -> str:
        return str(self)

    def get_n_agents(self) -> int:
        return self.agent_perm.get_n_agents()

    def get_agent_perm(self) -> AgentPerm:
        return self.agent_perm
    
    def get_io_perm(self) -> IOPerm:
        return self.io_perm

    def agent_is_trivial(self,i_agent: int) -> bool:
        """Return True if the agent is mapped to itself and its input & output are unchanged"""
        return self.get_agent_perm().agent_is_trivial(i_agent) and self.get_io_perm().agent_is_trivial(i_agent)

    def event_is_trivial(self,i_agent: int, setting: int, outcome: int) -> bool:
        """Return True if the agent is mapped to itself and the particular setting & outcome are unchanged"""
        return (self.get_agent_perm()[i_agent] == i_agent) and self.get_io_perm().event_is_trivial(i_agent,setting,outcome)

    def get_initial_state(self) -> List[Tuple[int,int,int]]:
        """Gives a pretty representation of the initial state of events before symmetry"""
        io_perm = self.get_io_perm()
        
        # This has the structure of [item for sublist in l for item in sublist] to efficiently flatten out
        return [
                    (1+i_agent, setting, outcome)
            for i_agent in range(self.get_n_agents()) if not self.agent_is_trivial(i_agent)
        for setting,outcome in itertools.product( io_perm.get_icard_iter(i_agent), io_perm.get_ocard_iter(i_agent) ) 
        if not self.event_is_trivial(i_agent, setting, outcome)
        ]

    def get_final_state(self) -> List[Tuple[int,int,int]]:
        """Gives a pretty representation of the final state of events after symmetry"""
        io_perm = self.get_io_perm()
        return [
                (1+self.get_agent_perm()[i_agent], io_perm.get_iperm(i_agent)[setting], io_perm.get_operm(i_agent)[outcome])
            for i_agent in range(self.get_n_agents()) if not self.agent_is_trivial(i_agent)
        for setting,outcome in itertools.product( io_perm.get_icard_iter(i_agent), io_perm.get_ocard_iter(i_agent) )
        if not self.event_is_trivial(i_agent, setting, outcome)
        ]
            
class DistrSym:
    """A class that stores an AgentIOPerm and a SourcePerm."""

    def __init__(self,agent_io_perm: AgentIOPerm,source_perm: SourcePerm):
        self.agent_io_perm = agent_io_perm
        self.source_perm = source_perm

    def to_data(self) -> Tuple[List[Tuple[int,int,int]], List[Tuple[int,int,int]], List[int]]:
        """A convenient data structure for later inflation purposes
        This functions returns
        (
            # Initial state
            [
                (agent_id,x,a) # agent_id = 1,2,..., x = setting = 0,1,..., a = outcome = 0,1,...
                ...
            ]
            # Output state under symmetry
            [
                g[agent_id,x,a]
                ...
            ]
            # Output source state under symmetry
            [
                g[0] # image of source s0 under symmetry
                ...
            ]
        )"""

        return self.agent_io_perm.get_initial_state(), \
               self.agent_io_perm.get_final_state(),   \
               self.source_perm.get_array()

    def state_to_str(self, state: List[Tuple[int,int,int]]) -> str:
        """Utility for __str__(self)"""
        ret = ""
        for i, the_tuple in enumerate(state):
            i_agent, setting, outcome = the_tuple
            ret += INT_TO_NAME[i_agent-1] + "(" + str(outcome) + "|" + str(setting) + ")"
            if i < len(state)-1:
                ret += " "
        return ret

    def __str__(self) -> str:
        """A nice string representation of a symmetry, based on to_data(self)"""
        initial_state, final_state, source_perm_array = self.to_data()

        return "[Sym] Events: " + self.state_to_str(initial_state) + "\n" \
             + "           to " + self.state_to_str(final_state)   + "\n" \
             + "     Sources: " + str(["S"+str(i) for i in range(len(source_perm_array)) 
                                                  if not self.source_perm.source_is_trivial(i)]) + "\n" \
             + "           to " + str(["S"+str(i) for i in source_perm_array
                                                  if not self.source_perm.source_is_trivial(i)])

    def to_ugly_string(self) -> str:
        """A low-level one-line string to represent a symmetry"""
        return "[Sym][Ugly] " + str(self.agent_io_perm) + " sources " + str(self.source_perm)

# -----------------
# -- GRAPH CLASS --
# -----------------

class NetworkGraph:
    """A network graph, stored as a Boolean matrix indicating the connectivity
    from sources (rows) to agents (columns)"""

    def __init__(self,n_sources:  int = None,
                      n_agents:   int = None,
                      shape: List[int]= None):
        if shape is None:
            shape = [n_sources,n_agents]
            
        self.mat = np.ndarray(shape,dtype=bool)
        self.mat.fill(False)

    def __setitem__(self,indices: Tuple[int,int],val: bool):
        """indices should be a pair (i_source,i_agent)"""
        self.mat[indices] = val
    
    def __getitem__(self,indices: Tuple[int,int]) -> bool:
        """indices shoudl be a pair (i_source,i_agent)"""
        return self.mat[indices]
        
    def __eq__(self, other: "NetworkGraph") -> bool:
        if isinstance(other, NetworkGraph):
            return np.array_equal(self.mat, other.mat)
        else:
            return NotImplemented

    def get_n_sources(self) -> int:
        return self.mat.shape[0]
    
    def get_n_agents(self) -> int:
        return self.mat.shape[1]

    def get_shape(self) -> Tuple[int,int]:
        return self.mat.shape
    
    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        ret  = "----- GRAPH -----\n"
        for i_source in range(self.get_n_sources()):
            ret += "Source " + INT_TO_SOURCE[i_source] + " is connected to agents "

            first_agent_found = False
            for i_agent in range(self.get_n_agents()):

                if self[i_source,i_agent]:
                    if first_agent_found:
                        ret += ", "
                    first_agent_found = True

                    ret += INT_TO_NAME[i_agent]

            ret += "\n"
        ret += "-----------------"
        return ret

    def is_automorphism(self,agent_perm: AgentPerm) -> Tuple[bool,SourcePerm]:
        """
        #      A                   B                B
        #   
        #   s3   s1      ->    s3     s1  ->   s1     s2
        #
        # C    s2   B        A    s2    C     A     s3   C
        #
        # i.e., agent_perm = [1,2,0] on the graph
        # g =   [ 1 1 0,
        #         0 1 1,
        #         1 0 1]
        # yields
        # new_g=[ 0 1 1   s1_old -> s2
        #         1 0 1   s2_old -> s3
        #         1 1 0 ] s3_old -> s1
        # This function returns (True,a source permutation) if the agent permutation
        # induces a graph automorphism (i.e., can be completed to a graph
        # automorphism by an extra permutation of the sources, which is to be returned),
        # or else returns (False,[])
        """
        new_g = NetworkGraph(shape=self.get_shape())
    
        # Act on the agents of g with agent_perm
        for i_source in range(self.get_n_sources()):
            for i_agent in range(self.get_n_agents()):
                # g[s,p] -> sigma.g[s,p] = g[s,sigma^{-1}(p)]
                new_g[i_source,agent_perm[i_agent]] = self[i_source,i_agent]

        all_source_perms = [SourcePerm(bare_perm) for bare_perm in itertools.permutations(range(self.get_n_sources()))]

        # Now try to act on the new_g (actively) to go back to the old_g
        for source_perm in all_source_perms:
            new_g_perm = NetworkGraph(shape=self.get_shape())

            for i_source in range(self.get_n_sources()):
                for i_agent in range(self.get_n_agents()):
                    new_g_perm[source_perm[i_source], i_agent] = new_g[i_source, i_agent]

            if new_g_perm == self:
                return (True,source_perm)

        return (False, SourcePerm([0]))

    def get_automorphisms(self) -> List[Tuple[AgentPerm,SourcePerm]]:
        """This function returns a list of tuples (agent perm,source perm) that are automorphisms of the graph.
        Tested only on bilocal and triangle."""

        # Nb: this includes the identity permutation
        all_agent_perms = [AgentPerm(bare_perm) for bare_perm in itertools.permutations(range(self.get_n_agents()))]

        automorphisms = []
        for agent_perm in all_agent_perms:
            # Here check if agent_perm works
            is_valid,source_perm = self.is_automorphism(agent_perm)

            if is_valid:
                automorphisms.append( (agent_perm, source_perm) )

        return automorphisms

# -----------------
# -- DISTR CLASS --
# -----------------

class Distr:
    """A conditional distribution, stored as a p[a][b][c][x][y][z] array"""

    def __init__(self,cardinalities: List[int]):
        self.mat = np.ndarray(cardinalities)
        self.mat.fill(0.)
        self.n_agents = int(len(self.mat.shape)/2)

    def __setitem__(self,indices: List[int],val: float):
        self.mat[indices] = val
    
    def __getitem__(self,indices: List[int]) -> float:
        return self.mat[indices]

    def get_n_agents(self) -> int:
        return self.n_agents

    def get_shape(self) -> List[int]:
        return self.mat.shape

    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str: 
        nonzero_elements = np.nonzero(self.mat)
        nonzero_elements_values = self.mat[nonzero_elements]
        
        ret  = "----- DISTR -----\n"
        for i,p_i in enumerate(nonzero_elements_values):
            for i_agent in range(self.get_n_agents()):
                ret += INT_TO_NAME[i_agent] + " = " + str(nonzero_elements[i_agent][i])
                if i_agent < self.get_n_agents()-1:
                    ret += ", "
            ret += " | "
            for i_agent in range(self.get_n_agents()):
                ret += INT_TO_SETTING[i_agent] + " = " + str(nonzero_elements[i_agent+self.get_n_agents()][i])
                if i_agent < self.get_n_agents()-1:
                    ret += ", "
            ret += " = " + str(p_i) + "\n"
        ret += "-----------------"

        return ret

    def is_symmetry(self,agent_io_perm: AgentIOPerm) -> bool:
        """def output_input_agent_perm_is_ok(p,output_input_perm,agent_perm):
        Note: there should be some preliminary check of whether the agent perm is a graph symmetry!
        Input: a distribution p, and a list of permutations for each dimension of p,
                and a permutation of the agents
        Returns: True if io_perm . agent_perm[p] = p (in this order)
        So on components, want p[a] = p[agent_perm^{-1}.io_perm^{-1}[a]]
        or equivalently p[io_perm.agent_perm[a]] = p[a]"""

        n_agents = self.get_n_agents()

        agent_perm = agent_io_perm.get_agent_perm()
        io_perm = agent_io_perm.get_io_perm()

        for indices in np.ndindex(self.get_shape()):

            # these will be agent_perm . outcome_perm
            perm_indices = [0] * len(indices)

            # FIRST the agent_perm

            # perm_indices = agent_perm[indices]
            for i_agent in range(n_agents):
                perm_indices[         agent_perm[i_agent]] = indices[         i_agent]
                perm_indices[n_agents+agent_perm[i_agent]] = indices[n_agents+i_agent]
            
            # THEN the io_perm
            # perm_indices = output_input_perm[perm_indices]
            for i, the_perm in enumerate(io_perm):
                perm_indices[i] = the_perm[perm_indices[i]]
            
            # Important to access the elements of the ndarray new_p...
            perm_indices = tuple(perm_indices)

            if not math.isclose(self[perm_indices],self[indices]):
                return False
        
        return True

    def get_symmetries(self,g: NetworkGraph) -> List[DistrSym]:
        """old: find_distrib_symmetries(g,p)."""
        assert g.get_n_agents() == self.get_n_agents()

        graph_automorphisms = g.get_automorphisms()

        # This part generates all AgentIOPerm's relevant to the current case.

        union_of_all_io_perms = [ list(itertools.permutations(range(self.get_shape()[i]))) for i in range(2*self.get_n_agents()) ]
        all_io_perms = ( IOPerm(bare_io_perm) for bare_io_perm in itertools.product(*union_of_all_io_perms) )

        all_agent_io_source_perms = (
                         # put 0 for the graph_auto and 0 for AgentPerm
            (AgentIOPerm( joint_graphauto_and_ioperm[0][0], joint_graphauto_and_ioperm[1] ),
                          joint_graphauto_and_ioperm[0][1] )
            for joint_graphauto_and_ioperm in itertools.product(graph_automorphisms,all_io_perms) )

        return [
                DistrSym(agent_io_perm,source_perm) 
            for agent_io_perm,source_perm in all_agent_io_source_perms 
            if self.is_symmetry(agent_io_perm)]
