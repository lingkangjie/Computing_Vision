#ifndef CAFFE_UTIL_DB_HPP
#define CAFFE_UTIL_DB_HPP

#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe { namespace db {

// Each enumerator becomes a named constant of
// the enumeration's type (that is, name)
// Mode m = READ; //m is a variable, OK
enum Mode { READ, WRITE, NEW };

class Cursor {
 public:
  Cursor() { }
  virtual ~Cursor() { }
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual string key() = 0;
  virtual string value() = 0;
  virtual bool valid() = 0;

  DISABLE_COPY_AND_ASSIGN(Cursor);
};

class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  virtual void Put(const string& key, const string& value) = 0;
  virtual void Commit() = 0;

  DISABLE_COPY_AND_ASSIGN(Transaction);
};

class DB {
 public:
  DB() { }
  virtual ~DB() { }
  virtual void Open(const string& source, Mode mode) = 0;
  virtual void Close() = 0;
  virtual Cursor* NewCursor() = 0;
  virtual Transaction* NewTransaction() = 0;

  DISABLE_COPY_AND_ASSIGN(DB);
};

/* in caffe.proto
 * message DataParameter{
 * enum DB { LEVELDB=0;LMDB=1;}
 * ...}
 * DB* GetDB is a stand-along function, NOT belongs to DB class.
 * But its return type is relationship with DB class.
 * in db.cpp file, we will implemente these two function, to get
 * which of database we want, LevelDB? or LDMB? As db_leveldb.hpp
 * and db_lmdb.hpp 'inherits' db.hpp, that is to say, LevelDB and LMDB
 * class inherits DB class, so the return type DB* of GetDB() function
 * absolutely has ability to point to DB() or LevelDB() or LMDB() by
 * type upcast and downcast
 */

DB* GetDB(DataParameter::DB backend);
DB* GetDB(const string& backend);

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_HPP
