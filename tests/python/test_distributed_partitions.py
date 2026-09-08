"""In-process partition/halo map contracts: no torch, GPU, or MPI required."""

from __future__ import annotations

import struct
import unittest

from tiga.distributed import (
    ReceivedHalo, collective_halo_maps, derive_halo_map, destination_owner,
    owned_range, pack_halo, unpack_halo,
)


class DistributedPartitionTest(unittest.TestCase):
    def test_collective_halo_maps_uneven_partition_is_symmetric(self):
        entities, world_size = 10, 3  # remainder path: 4/3/3 owned split
        row_ptr = [0]
        col_idx = []
        for destination in range(entities):
            degree = destination % 3 + 1
            for neighbor in range(degree):
                col_idx.append((destination * 3 + neighbor * 4 + 1) % entities)
            row_ptr.append(len(col_idx))
        halos = collective_halo_maps(
            row_ptr, col_idx, num_entities=entities, world_size=world_size)
        self.assertEqual(len(halos), world_size)

        # owned_range tiles [0, entities) without gaps or overlap.
        covered = []
        for rank, halo in enumerate(halos):
            begin, end = owned_range(entities, world_size, rank)
            self.assertEqual((halo.owned_begin, halo.owned_end), (begin, end))
            self.assertEqual(halo.owned_entities, end - begin)
            covered.extend(range(begin, end))
        self.assertEqual(covered, list(range(entities)))

        # Every send is mirrored by the peer's receive, and every ghost id
        # belongs to the rank that sends it.
        for rank, halo in enumerate(halos):
            for peer, ids in halo.send_to:
                self.assertEqual(dict(halos[peer].receive_from)[rank], ids)
                self.assertTrue(all(
                    destination_owner(entity, entities, world_size) == rank
                    for entity in ids))
            for peer, ids in halo.receive_from:
                self.assertEqual(dict(halos[peer].send_to)[rank], ids)
                self.assertTrue(all(
                    destination_owner(entity, entities, world_size) == peer
                    for entity in ids))
            self.assertTrue(all(
                not halo.owned_begin <= ghost < halo.owned_end
                for ghost in halo.ghost_ids))

    def test_derive_halo_map_validates_topology(self):
        with self.assertRaisesRegex(ValueError, "monotonic"):
            derive_halo_map(
                [0, 2, 1, 3], [0, 1, 2],
                num_entities=3, world_size=2, rank=0)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            derive_halo_map(
                [0, 1, 2, 3], [0, 1, 5],
                num_entities=3, world_size=2, rank=0)
        with self.assertRaisesRegex(ValueError, "not owned"):
            derive_halo_map(
                [0, 1, 2, 3], [0, 1, 2],
                num_entities=3, world_size=2, rank=0,
                peer_requests=((), (2,)))

    def test_pack_unpack_round_trip_without_transport(self):
        # Rank zero owns [0, 3); its rows reference ghosts 3, 4, 5 and peer 1
        # requests the non-contiguous owned ids 0 and 2, which forces the
        # gather branch in pack_halo.
        row_ptr = [0, 2, 4, 5, 5, 6, 6]
        col_idx = [1, 4, 0, 5, 3, 2]
        halo = derive_halo_map(
            row_ptr, col_idx, num_entities=6, world_size=2, rank=0,
            peer_requests=((), (0, 2)))
        self.assertEqual(halo.ghost_ids, (3, 4, 5))
        self.assertEqual(halo.send_to, ((1, (0, 2)),))

        element_bytes = 8
        owned = b"".join(
            struct.pack("q", entity * 10)
            for entity in range(halo.owned_begin, halo.owned_end))
        packed = pack_halo(halo, owned, element_bytes=element_bytes)
        ((peer, ids, payload),) = packed.messages
        self.assertEqual((peer, ids), (1, (0, 2)))
        self.assertEqual(struct.unpack("2q", payload), (0, 20))

        received = ReceivedHalo(
            tuple(
                (entity, struct.pack("q", entity * 100))
                for entity in halo.ghost_ids),
            element_bytes,
        )
        ghosts = unpack_halo(halo, received)
        self.assertEqual(ghosts.ids, halo.ghost_ids)
        for entity in halo.ghost_ids:
            self.assertEqual(
                struct.unpack("q", ghosts.value_bytes(entity))[0],
                entity * 100)

        packed_received = ReceivedHalo(
            (), element_bytes,
            packed_ids=halo.ghost_ids,
            packed_data=b"".join(
                struct.pack("q", entity * 100) for entity in halo.ghost_ids),
        )
        self.assertEqual(unpack_halo(halo, packed_received), ghosts)


if __name__ == "__main__":
    unittest.main()
