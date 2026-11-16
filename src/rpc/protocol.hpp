// Wire protocol of the query service: one request line, one response line, over TCP.
//
//   PING                 ->  OK pong
//   STATS                ->  OK n=<pixels per channel> mean=<r>,<g>,<b> std=<r>,<g>,<b>
//   CLASS <class_id>     ->  OK class=<k> n=<..> mean=<r>,<g>,<b> std=<r>,<g>,<b>
//   IMAGE <image_id>     ->  OK image=<id> class=<k> split=train|test
//                               crc_resized=<8 hex> crc_normalized=<8 hex>
//   anything else        ->  ERR <message>
//
// Every operation is a read of immutable results, so it is idempotent: a client that timed
// out may safely send the same request again.
#pragma once

namespace rpc {
constexpr int kDefaultPort = 4010;
constexpr int kMaxLine = 512;
}  // namespace rpc
