"""Connectome reconstructions as an empirical test bed for the quotient work.

The RNN results all rest on a partition we *inferred* (signed WL on W_rec). The
fly central complex offers the one thing that setup never had: an externally
defined, anatomy-derived candidate partition (columnar identity -- PB glomerulus
/ EB wedge), fixed before anyone looked at the connectivity. That turns
"is this partition equitable?" from a search into a hypothesis test.

Two reconstructions are pulled independently:

  hemibrain  -- Janelia, one hemisphere, dense proofread EM. Unsigned: no
                neurotransmitter calls, so only `gamma_ep` applies.
  FlyWire    -- FAFB, both hemispheres, plus per-synapse neurotransmitter
                predictions (Eckstein et al. 2024). Signing the graph by
                predicted NT makes `gamma_signed` applicable -- the same
                signed/unsigned pair we use on W_rec.

Note on independence: these are different animals and different pipelines, but
CX cell-type nomenclature is shared lineage (hemibrain typing was carried into
FlyWire annotation). Agreement on *partition geometry* is real evidence;
agreement on *type labels* is partly circular. Keep the claim on the former.
"""
